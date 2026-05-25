package wrsllm

import "errors"

// maxErrorBodyBytes caps how much of a non-2xx upstream response body is read
// into a returned error, so a huge or hostile error body can't bloat memory or
// the error string. The body never carries the API key.
const maxErrorBodyBytes = 4096

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

	// ErrEmptyCompletion is returned when a provider receives HTTP 200 but the
	// upstream body is semantically empty (no generated text). A blank
	// generation is treated as a failure, not a silent success — if an empty
	// completion is ever legitimate, that must be an explicit opt-in.
	ErrEmptyCompletion = errors.New("wrsllm: provider returned an empty completion")

	// ErrInvalidConfig is returned by RouterConfig.Validate when the config is
	// structurally unable to route any request.
	ErrInvalidConfig = errors.New("wrsllm: invalid router config")
)
