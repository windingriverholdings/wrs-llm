package wrsllm

import "fmt"

// KillSwitch is a global override that forces all routing to a single tier.
type KillSwitch int

const (
	// KillSwitchOff leaves normal routing in place (default).
	KillSwitchOff KillSwitch = iota
	// ForceAllLocal forces every request to the local provider.
	ForceAllLocal
	// ForceAllClaude forces every request to the cloud provider, EXCEPT
	// Sensitive requests, which remain hard-guarded to local.
	ForceAllClaude
)

// Route is a resolved {provider, model} pair.
type Route struct {
	Provider Provider
	Model    string
}

// RouterConfig is injected by the caller (constructed from its own env). The
// module holds NO package-level global state.
type RouterConfig struct {
	// Local is the local (Ollama) provider. Required for Sensitive routing and
	// ForceAllLocal.
	Local Provider
	// Cloud is the cloud (Claude) provider. Required for ForceAllClaude and
	// cloud-targeted overrides.
	Cloud Provider

	// Routes maps each TaskKind to its default {provider, model}.
	Routes map[TaskKind]Route

	// Providers maps a provider name (e.g. "local-ollama") to a Provider, used
	// to resolve Override by provider name.
	Providers map[string]Provider

	// Models maps a model key (e.g. "claude-sonnet") to a {provider, model},
	// used to resolve Override by model key.
	Models map[string]Route

	// KillSwitch, when set, forces all routing to one tier (see KillSwitch).
	KillSwitch KillSwitch
}

// RouteRequest describes a single routing decision input.
type RouteRequest struct {
	TaskKind   TaskKind
	Text       string // for future length/heuristic classification
	Sensitive  bool   // PII/private -> FORCED local, never cloud
	NeedsTools bool   // agentic -> REFUSED (single-shot only)
	Override   string // explicit model key or provider name; highest precedence after hard guards
	LowConf    bool   // escalation flag; reserved for the future classifier seam
}

// RouteDecision is the resolved routing outcome. Reason is always populated for
// cost-audit logging.
type RouteDecision struct {
	Provider Provider

	// Model is ADVISORY. The providers in this module are pre-bound to a model
	// at construction (NewOllamaProvider / NewClaudeProvider) and send that
	// construction-time model on the wire, NOT this field. Model is here for
	// logging, cost-audit, and callers that resolve a provider+model pair
	// themselves. On the forced-provider paths (sensitive, kill-switch,
	// override-by-provider-name) Model may be "" when the task-kind route points
	// at a different provider — see modelForTaskKind.
	Model string

	Reason string
}

// Router resolves RouteRequests to RouteDecisions under a layered policy.
type Router struct {
	cfg RouterConfig
}

// Validate reports whether the config is structurally capable of routing. It
// fails LOUD (returns a wrapped ErrInvalidConfig) when:
//   - the config has no routing surface at all (no Routes, no Local, no Cloud),
//     so every request would dead-end; or
//   - any Routes entry carries a nil Provider, which would nil-panic at the
//     caller's Generate.
func (c RouterConfig) Validate() error {
	if len(c.Routes) == 0 && c.Local == nil && c.Cloud == nil {
		return fmt.Errorf("%w: no routes, no local provider, and no cloud provider configured", ErrInvalidConfig)
	}
	for kind, route := range c.Routes {
		if route.Provider == nil {
			return fmt.Errorf("%w: route for %s has a nil provider", ErrInvalidConfig, kind)
		}
	}
	return nil
}

// NewRouter constructs a Router from an injected config. It does NOT validate;
// callers that want construction-time validation should use NewRouterValidated.
func NewRouter(cfg RouterConfig) *Router {
	return &Router{cfg: cfg}
}

// NewRouterValidated constructs a Router and fails LOUD if the config is
// structurally broken (see RouterConfig.Validate).
func NewRouterValidated(cfg RouterConfig) (*Router, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &Router{cfg: cfg}, nil
}

// Route applies the layered policy, highest precedence first:
//
//  1. Hard guard: NeedsTools -> ErrAgenticNotSupported (single-shot only).
//  2. Hard guard: Sensitive -> force local; ErrNoLocalProvider if none.
//     This beats overrides AND the kill switch — PII never leaves the box.
//  3. Override: explicit model key or provider name.
//  4. Kill switch: ForceAllLocal / ForceAllClaude.
//  5. Deterministic TaskKind -> {provider, model} map.
//  6. (Future) classifier seam — not built; see LowConf.
//
// Every returned decision sets Reason for cost-audit logging.
func (r *Router) Route(req RouteRequest) (RouteDecision, error) {
	// 1. Agentic refusal — absolute, beats everything.
	if req.NeedsTools {
		return RouteDecision{}, ErrAgenticNotSupported
	}

	// 2. Sensitive hard guard — force local, beats override + kill switch.
	if req.Sensitive {
		if r.cfg.Local == nil {
			return RouteDecision{}, ErrNoLocalProvider
		}
		return RouteDecision{
			Provider: r.cfg.Local,
			Model:    r.modelForTaskKind(req.TaskKind, r.cfg.Local),
			Reason:   fmt.Sprintf("sensitive=true forced local for %s", req.TaskKind),
		}, nil
	}

	// 3. Explicit override.
	if req.Override != "" {
		if route, ok := r.cfg.Models[req.Override]; ok {
			return RouteDecision{
				Provider: route.Provider,
				Model:    route.Model,
				Reason:   fmt.Sprintf("override model key %q", req.Override),
			}, nil
		}
		if p, ok := r.cfg.Providers[req.Override]; ok {
			return RouteDecision{
				Provider: p,
				Model:    r.modelForTaskKind(req.TaskKind, p),
				Reason:   fmt.Sprintf("override provider %q", req.Override),
			}, nil
		}
		return RouteDecision{}, fmt.Errorf("%w: %q", ErrUnknownOverride, req.Override)
	}

	// 4. Kill switch.
	switch r.cfg.KillSwitch {
	case ForceAllLocal:
		if r.cfg.Local == nil {
			return RouteDecision{}, ErrNoLocalProvider
		}
		return RouteDecision{
			Provider: r.cfg.Local,
			Model:    r.modelForTaskKind(req.TaskKind, r.cfg.Local),
			Reason:   fmt.Sprintf("kill-switch force-all-local for %s", req.TaskKind),
		}, nil
	case ForceAllClaude:
		if r.cfg.Cloud == nil {
			return RouteDecision{}, ErrNoCloudProvider
		}
		return RouteDecision{
			Provider: r.cfg.Cloud,
			Model:    r.modelForTaskKind(req.TaskKind, r.cfg.Cloud),
			Reason:   fmt.Sprintf("kill-switch force-all-claude for %s", req.TaskKind),
		}, nil
	}

	// 5. Deterministic task-kind map.
	if route, ok := r.cfg.Routes[req.TaskKind]; ok {
		// Guard against a matched route with a nil provider: returning it as a
		// success would nil-panic at the caller's Generate. Fail LOUD instead.
		if route.Provider == nil {
			return RouteDecision{}, fmt.Errorf("%w: route for %s has nil provider", ErrNoRoute, req.TaskKind)
		}
		return RouteDecision{
			Provider: route.Provider,
			Model:    route.Model,
			Reason:   fmt.Sprintf("task-kind map %s -> %s", req.TaskKind, route.Model),
		}, nil
	}

	// 6. Future classifier seam would slot here (gated on LowConf). Not built.
	return RouteDecision{}, fmt.Errorf("%w: %s", ErrNoRoute, req.TaskKind)
}

// modelForTaskKind returns the model configured for the task kind IF the
// configured route uses the same provider; otherwise it returns "" so the
// caller (override / kill-switch / sensitive paths) supplies a forced provider
// without inheriting a mismatched model name. The provider's own default model
// (set at construction) governs when the model string is empty.
func (r *Router) modelForTaskKind(kind TaskKind, provider Provider) string {
	if route, ok := r.cfg.Routes[kind]; ok && route.Provider == provider {
		return route.Model
	}
	return ""
}
