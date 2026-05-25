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
	Model    string
	Reason   string
}

// Router resolves RouteRequests to RouteDecisions under a layered policy.
type Router struct {
	cfg RouterConfig
}

// NewRouter constructs a Router from an injected config.
func NewRouter(cfg RouterConfig) *Router {
	return &Router{cfg: cfg}
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
