package wrsllm

import "errors"

// Typed routing errors. Callers should branch with errors.Is.
var (
	// ErrAgenticNotSupported is returned when a request sets NeedsTools.
	// This module is single-shot only; agentic/tool-using work belongs to a
	// coding agent such as `claude -p`.
	ErrAgenticNotSupported = errors.New("wrsllm: agentic (tool-using) requests are not supported; this module is single-shot")

	// ErrNoLocalProvider is returned when a Sensitive request must be forced
	// local but the RouterConfig has no local provider wired.
	ErrNoLocalProvider = errors.New("wrsllm: sensitive request requires a local provider but none is configured")

	// ErrNoCloudProvider is returned when routing resolves to the cloud
	// provider (e.g. force-all-claude) but none is configured.
	ErrNoCloudProvider = errors.New("wrsllm: routing requires a cloud provider but none is configured")

	// ErrNoRoute is returned when a TaskKind has no configured route and no
	// higher-precedence rule applied.
	ErrNoRoute = errors.New("wrsllm: no route configured for task kind")

	// ErrUnknownOverride is returned when RouteRequest.Override names neither a
	// configured model key nor a configured provider name.
	ErrUnknownOverride = errors.New("wrsllm: override does not match any configured model or provider")
)
