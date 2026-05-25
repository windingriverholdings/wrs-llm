package wrsllm

import (
	"context"
	"errors"
	"testing"
)

// stubProvider is a no-op Provider used to assert routing decisions
// without making any network calls.
type stubProvider struct {
	id string
}

func (s *stubProvider) Generate(_ context.Context, _ string, _ string) (string, error) {
	return "", nil
}

// newTestConfig builds a RouterConfig with both a local (Ollama) and a
// cloud (Claude) provider wired for each TaskKind, mirroring the default
// strategy map.
func newTestConfig() RouterConfig {
	local := &stubProvider{id: "local-ollama"}
	cloud := &stubProvider{id: "cloud-claude"}
	return RouterConfig{
		Local: local,
		Cloud: cloud,
		Routes: map[TaskKind]Route{
			Extract:             {Provider: local, Model: "qwen3:8b"},
			PersonaCreation:     {Provider: local, Model: "qwen3:8b"},
			Summarize:           {Provider: local, Model: "llama3.1:8b"},
			ConversationalReply: {Provider: local, Model: "llama3.1:8b"},
			Caption:             {Provider: local, Model: "llama3.1:8b"},
		},
		Providers: map[string]Provider{
			"local-ollama": local,
			"cloud-claude": cloud,
		},
		Models: map[string]Route{
			"claude-sonnet": {Provider: cloud, Model: "claude-sonnet-4"},
		},
	}
}

func TestRoute_NeedsToolsReturnsTypedError(t *testing.T) {
	r := NewRouter(newTestConfig())

	_, err := r.Route(RouteRequest{TaskKind: Extract, NeedsTools: true})
	if err == nil {
		t.Fatal("expected error for NeedsTools=true, got nil")
	}
	if !errors.Is(err, ErrAgenticNotSupported) {
		t.Fatalf("expected ErrAgenticNotSupported, got %v", err)
	}
}

func TestRoute_NeedsToolsReturnsNoProvider(t *testing.T) {
	r := NewRouter(newTestConfig())

	dec, _ := r.Route(RouteRequest{TaskKind: Extract, NeedsTools: true})
	if dec.Provider != nil {
		t.Fatalf("expected no provider on agentic refusal, got %v", dec.Provider)
	}
}

func TestRoute_SensitiveForcesLocal(t *testing.T) {
	cfg := newTestConfig()
	r := NewRouter(cfg)

	dec, err := r.Route(RouteRequest{TaskKind: ConversationalReply, Sensitive: true})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dec.Provider != cfg.Local {
		t.Fatalf("expected sensitive request forced to local provider, got %v", dec.Provider)
	}
	if dec.Reason == "" {
		t.Fatal("expected Reason populated for sensitive routing")
	}
}

func TestRoute_SensitiveNoLocalConfiguredReturnsTypedError(t *testing.T) {
	cfg := newTestConfig()
	cfg.Local = nil // no local provider available
	r := NewRouter(cfg)

	_, err := r.Route(RouteRequest{TaskKind: ConversationalReply, Sensitive: true})
	if err == nil {
		t.Fatal("expected error when sensitive request has no local provider")
	}
	if !errors.Is(err, ErrNoLocalProvider) {
		t.Fatalf("expected ErrNoLocalProvider, got %v", err)
	}
}

func TestRoute_SensitiveBeatsCloudOverride(t *testing.T) {
	// Sensitive (hard guard) must win even when an explicit override
	// names a cloud model — PII never leaves the box.
	cfg := newTestConfig()
	r := NewRouter(cfg)

	dec, err := r.Route(RouteRequest{
		TaskKind:  Summarize,
		Sensitive: true,
		Override:  "claude-sonnet",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dec.Provider != cfg.Local {
		t.Fatal("sensitive must force local even with a cloud override")
	}
}

func TestRoute_TaskKindMapsToConfiguredProviderAndModel(t *testing.T) {
	cfg := newTestConfig()
	r := NewRouter(cfg)

	cases := []struct {
		kind  TaskKind
		model string
	}{
		{Extract, "qwen3:8b"},
		{PersonaCreation, "qwen3:8b"},
		{Summarize, "llama3.1:8b"},
		{ConversationalReply, "llama3.1:8b"},
		{Caption, "llama3.1:8b"},
	}

	for _, c := range cases {
		dec, err := r.Route(RouteRequest{TaskKind: c.kind})
		if err != nil {
			t.Fatalf("kind %v: unexpected error: %v", c.kind, err)
		}
		if dec.Model != c.model {
			t.Errorf("kind %v: expected model %q, got %q", c.kind, c.model, dec.Model)
		}
		if dec.Provider != cfg.Local {
			t.Errorf("kind %v: expected local provider", c.kind)
		}
		if dec.Reason == "" {
			t.Errorf("kind %v: expected Reason populated", c.kind)
		}
	}
}

func TestRoute_UnknownTaskKindReturnsTypedError(t *testing.T) {
	cfg := newTestConfig()
	delete(cfg.Routes, Caption)
	r := NewRouter(cfg)

	_, err := r.Route(RouteRequest{TaskKind: Caption})
	if !errors.Is(err, ErrNoRoute) {
		t.Fatalf("expected ErrNoRoute for unmapped task kind, got %v", err)
	}
}

func TestRoute_OverrideByModelKey(t *testing.T) {
	cfg := newTestConfig()
	r := NewRouter(cfg)

	dec, err := r.Route(RouteRequest{TaskKind: Extract, Override: "claude-sonnet"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dec.Provider != cfg.Cloud {
		t.Fatal("expected override to route to cloud provider")
	}
	if dec.Model != "claude-sonnet-4" {
		t.Fatalf("expected override model claude-sonnet-4, got %q", dec.Model)
	}
}

func TestRoute_OverrideByProviderName(t *testing.T) {
	cfg := newTestConfig()
	r := NewRouter(cfg)

	dec, err := r.Route(RouteRequest{TaskKind: Extract, Override: "cloud-claude"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dec.Provider != cfg.Cloud {
		t.Fatal("expected override by provider name to route to cloud")
	}
}

func TestRoute_UnknownOverrideReturnsTypedError(t *testing.T) {
	cfg := newTestConfig()
	r := NewRouter(cfg)

	_, err := r.Route(RouteRequest{TaskKind: Extract, Override: "does-not-exist"})
	if !errors.Is(err, ErrUnknownOverride) {
		t.Fatalf("expected ErrUnknownOverride, got %v", err)
	}
}

func TestRoute_KillSwitchForceAllLocal(t *testing.T) {
	cfg := newTestConfig()
	cfg.KillSwitch = ForceAllLocal
	r := NewRouter(cfg)

	// Even a TaskKind that maps to local already — assert local + reason.
	dec, err := r.Route(RouteRequest{TaskKind: Summarize})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dec.Provider != cfg.Local {
		t.Fatal("force-all-local must route to local provider")
	}
	if dec.Reason == "" {
		t.Fatal("expected Reason populated for kill-switch routing")
	}
}

func TestRoute_KillSwitchForceAllClaude(t *testing.T) {
	cfg := newTestConfig()
	cfg.KillSwitch = ForceAllClaude
	r := NewRouter(cfg)

	dec, err := r.Route(RouteRequest{TaskKind: Summarize})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dec.Provider != cfg.Cloud {
		t.Fatal("force-all-claude must route to cloud provider")
	}
}

func TestRoute_KillSwitchForceAllClaudeNoCloudReturnsError(t *testing.T) {
	cfg := newTestConfig()
	cfg.Cloud = nil
	cfg.KillSwitch = ForceAllClaude
	r := NewRouter(cfg)

	_, err := r.Route(RouteRequest{TaskKind: Summarize})
	if !errors.Is(err, ErrNoCloudProvider) {
		t.Fatalf("expected ErrNoCloudProvider, got %v", err)
	}
}

func TestRoute_SensitiveBeatsForceAllClaudeKillSwitch(t *testing.T) {
	// Sensitive is a hard guard and must beat the force-all-claude
	// kill switch — PII must never go to cloud regardless of config.
	cfg := newTestConfig()
	cfg.KillSwitch = ForceAllClaude
	r := NewRouter(cfg)

	dec, err := r.Route(RouteRequest{TaskKind: Summarize, Sensitive: true})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dec.Provider != cfg.Local {
		t.Fatal("sensitive must beat force-all-claude kill switch")
	}
}

func TestRoute_NeedsToolsBeatsEverything(t *testing.T) {
	// Agentic refusal is absolute — even with an override and kill switch set.
	cfg := newTestConfig()
	cfg.KillSwitch = ForceAllLocal
	r := NewRouter(cfg)

	_, err := r.Route(RouteRequest{
		TaskKind:   Extract,
		NeedsTools: true,
		Override:   "claude-sonnet",
	})
	if !errors.Is(err, ErrAgenticNotSupported) {
		t.Fatalf("expected agentic refusal to beat all, got %v", err)
	}
}
