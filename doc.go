// Package wrsllm is the shared single-shot LLM provider + routing library for
// Winding River Software. It exposes a small Provider abstraction over Ollama
// (local) and Claude (cloud), plus a config-injected Router that maps a
// TaskKind to a concrete provider+model under a layered policy.
//
// # Single-shot vs agentic boundary
//
// This module performs ONE prompt -> ONE response. It is deliberately NOT
// agentic: there is no tool-calling loop, no multi-turn planning, no retry
// orchestration. Requests that need tools (RouteRequest.NeedsTools) are
// REFUSED with ErrAgenticNotSupported — that work belongs to a coding agent
// such as `claude -p`, not to this library.
//
// # No global state
//
// Unlike the provider code it was lifted from, wrsllm carries NO package-level
// singleton and NO config.Get() coupling. The caller constructs a RouterConfig
// from its own environment and injects it via NewRouter. This makes the module
// importable by multiple projects without contending over shared globals.
package wrsllm

import "context"

// Provider generates text from a single prompt + system message.
// Implementations are single-shot: one request, one response, no tool loop.
type Provider interface {
	Generate(ctx context.Context, prompt, system string) (string, error)
}
