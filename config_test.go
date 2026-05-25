package wrsllm

import "testing"

func TestDefaultRoutes_StrategyMap(t *testing.T) {
	local := &stubProvider{id: "local"}
	routes := DefaultRoutes(local)

	cases := []struct {
		kind  TaskKind
		model string
	}{
		{Extract, DefaultQwenModel},
		{PersonaCreation, DefaultQwenModel},
		{Summarize, DefaultLlamaModel},
		{ConversationalReply, DefaultLlamaModel},
		{Caption, DefaultLlamaModel},
	}
	for _, c := range cases {
		route, ok := routes[c.kind]
		if !ok {
			t.Fatalf("missing route for %s", c.kind)
		}
		if route.Model != c.model {
			t.Errorf("%s: expected model %q, got %q", c.kind, c.model, route.Model)
		}
		if route.Provider != local {
			t.Errorf("%s: expected supplied local provider", c.kind)
		}
	}
}

func TestDefaultRoutes_UsableWithRouter(t *testing.T) {
	local := &stubProvider{id: "local"}
	r := NewRouter(RouterConfig{Local: local, Routes: DefaultRoutes(local)})

	dec, err := r.Route(RouteRequest{TaskKind: Extract})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dec.Model != DefaultQwenModel {
		t.Fatalf("expected %q, got %q", DefaultQwenModel, dec.Model)
	}
}
