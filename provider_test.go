package wrsllm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestNewOllamaProvider_Defaults(t *testing.T) {
	p := NewOllamaProvider("http://localhost:11434/", "qwen3:8b", 0)
	if p == nil {
		t.Fatal("expected provider, got nil")
	}
	if p.baseURL != "http://localhost:11434" {
		t.Fatalf("expected trailing slash trimmed, got %q", p.baseURL)
	}
	if p.model != "qwen3:8b" {
		t.Fatalf("expected model qwen3:8b, got %q", p.model)
	}
	// zero timeout falls back to a sane default
	if p.client.Timeout != defaultOllamaTimeout {
		t.Fatalf("expected default timeout %v, got %v", defaultOllamaTimeout, p.client.Timeout)
	}
}

func TestNewOllamaProvider_CustomTimeout(t *testing.T) {
	p := NewOllamaProvider("http://localhost:11434", "qwen3:8b", 5*time.Second)
	if p.client.Timeout != 5*time.Second {
		t.Fatalf("expected 5s timeout, got %v", p.client.Timeout)
	}
}

func TestOllamaProvider_Generate(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			t.Errorf("expected /api/chat, got %s", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		var req ollamaChatRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("bad request body: %v", err)
		}
		if req.Stream {
			t.Error("expected non-streaming request")
		}
		if req.Model != "qwen3:8b" {
			t.Errorf("expected model qwen3:8b, got %q", req.Model)
		}
		// system + user messages
		if len(req.Messages) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(req.Messages))
		}
		if req.Messages[0].Role != "system" || req.Messages[1].Role != "user" {
			t.Errorf("unexpected message roles: %+v", req.Messages)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(ollamaChatResponse{
			Message: ollamaChatMsg{Role: "assistant", Content: "hello back"},
		})
	}))
	defer srv.Close()

	p := NewOllamaProvider(srv.URL, "qwen3:8b", 5*time.Second)
	out, err := p.Generate(context.Background(), "hello", "be terse")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out != "hello back" {
		t.Fatalf("expected 'hello back', got %q", out)
	}
}

func TestOllamaProvider_GenerateNoSystem(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req ollamaChatRequest
		_ = json.Unmarshal(body, &req)
		if len(req.Messages) != 1 {
			t.Fatalf("expected 1 message when system empty, got %d", len(req.Messages))
		}
		_ = json.NewEncoder(w).Encode(ollamaChatResponse{
			Message: ollamaChatMsg{Content: "ok"},
		})
	}))
	defer srv.Close()

	p := NewOllamaProvider(srv.URL, "qwen3:8b", 5*time.Second)
	if _, err := p.Generate(context.Background(), "hi", ""); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestOllamaProvider_GenerateNon200(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("boom"))
	}))
	defer srv.Close()

	p := NewOllamaProvider(srv.URL, "qwen3:8b", 5*time.Second)
	_, err := p.Generate(context.Background(), "hi", "")
	if err == nil {
		t.Fatal("expected error on non-200 status")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Fatalf("expected status code in error, got %v", err)
	}
}

func TestNewClaudeProvider_Defaults(t *testing.T) {
	p := NewClaudeProvider("sk-test", "claude-sonnet-4", 0)
	if p == nil {
		t.Fatal("expected provider, got nil")
	}
	if p.client.Timeout != defaultClaudeTimeout {
		t.Fatalf("expected default timeout %v, got %v", defaultClaudeTimeout, p.client.Timeout)
	}
}

func TestClaudeProvider_Generate(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "sk-test" {
			t.Errorf("expected x-api-key header, got %q", r.Header.Get("x-api-key"))
		}
		if r.Header.Get("anthropic-version") == "" {
			t.Error("expected anthropic-version header")
		}
		body, _ := io.ReadAll(r.Body)
		var req claudeRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("bad request body: %v", err)
		}
		if req.Model != "claude-sonnet-4" {
			t.Errorf("expected model claude-sonnet-4, got %q", req.Model)
		}
		if req.System != "be terse" {
			t.Errorf("expected system passed through, got %q", req.System)
		}
		w.Header().Set("Content-Type", "application/json")
		resp := claudeResponse{}
		resp.Content = []struct {
			Text string `json:"text"`
		}{{Text: "claude says hi"}}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	p := NewClaudeProvider("sk-test", "claude-sonnet-4", 5*time.Second)
	p.baseURL = srv.URL // override endpoint for test
	out, err := p.Generate(context.Background(), "hello", "be terse")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out != "claude says hi" {
		t.Fatalf("expected 'claude says hi', got %q", out)
	}
}

func TestClaudeProvider_GenerateNon200(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte("bad key"))
	}))
	defer srv.Close()

	p := NewClaudeProvider("sk-test", "claude-sonnet-4", 5*time.Second)
	p.baseURL = srv.URL
	_, err := p.Generate(context.Background(), "hi", "")
	if err == nil {
		t.Fatal("expected error on non-200")
	}
	if !strings.Contains(err.Error(), "401") {
		t.Fatalf("expected 401 in error, got %v", err)
	}
}
