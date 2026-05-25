package wrsllm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"
)

const (
	defaultClaudeTimeout   = 60 * time.Second
	defaultClaudeMaxTokens = 4096
	claudeAPIEndpoint      = "https://api.anthropic.com/v1/messages"
	anthropicVersion       = "2023-06-01"
)

// ClaudeProvider calls the Anthropic Messages API for single-shot text
// generation.
//
// Logic lifted from openbrain/internal/llm/claude.go and de-coupled from
// openbrain's config: apiKey, model, and timeout are explicit params. The
// baseURL field defaults to the live endpoint and is overridable for tests.
type ClaudeProvider struct {
	apiKey  string
	model   string
	baseURL string
	client  *http.Client
}

// NewClaudeProvider creates a Claude provider. A zero timeout falls back to
// defaultClaudeTimeout.
func NewClaudeProvider(apiKey, model string, timeout time.Duration) *ClaudeProvider {
	if timeout <= 0 {
		timeout = defaultClaudeTimeout
	}
	return &ClaudeProvider{
		apiKey:  apiKey,
		model:   model,
		baseURL: claudeAPIEndpoint,
		client:  &http.Client{Timeout: timeout},
	}
}

type claudeRequest struct {
	Model     string      `json:"model"`
	MaxTokens int         `json:"max_tokens"`
	System    string      `json:"system,omitempty"`
	Messages  []claudeMsg `json:"messages"`
}

type claudeMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type claudeResponse struct {
	Content []struct {
		Text string `json:"text"`
	} `json:"content"`
}

// Generate sends a single-message request to the Messages API and returns the
// first text block of the response.
func (p *ClaudeProvider) Generate(ctx context.Context, prompt, system string) (string, error) {
	reqBody := claudeRequest{
		Model:     p.model,
		MaxTokens: defaultClaudeMaxTokens,
		System:    system,
		Messages:  []claudeMsg{{Role: "user", Content: prompt}},
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal claude request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("create claude request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", anthropicVersion)

	resp, err := p.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("claude request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("claude: status %d: %s", resp.StatusCode, respBody)
	}

	var result claudeResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("decode claude response: %w", err)
	}

	content := ""
	if len(result.Content) > 0 {
		content = result.Content[0].Text
	}

	slog.Info("claude response", "model", p.model, "prompt_len", len(prompt), "response_len", len(content))
	return content, nil
}
