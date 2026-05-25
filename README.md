# wrs-llm

Shared **single-shot** LLM provider + routing library for Winding River Software.

`wrs-llm` is a small, dependency-free Go module that four WRS projects import to
decide *which model handles a given task* and to *run one prompt against it*. It
wraps two providers — **Ollama** (local) and **Claude** (cloud) — behind a tiny
interface, and adds a **config-injected router** that maps a task kind to a
concrete `{provider, model}` under a layered policy.

```
go get github.com/windingriverholdings/wrs-llm
```

```go
import wrsllm "github.com/windingriverholdings/wrs-llm"
```

---

## Single-shot vs agentic — the boundary

This module is **single-shot**: **one prompt → one response.** It is
deliberately **not** agentic.

| | single-shot (this module) | agentic |
|---|---|---|
| Shape | one prompt, one completion | tool-calling loop, multi-turn planning |
| Examples | extract facts, summarize, caption, reply | "fix this repo", "run the tests and iterate" |
| Belongs to | **wrs-llm** | a coding agent such as `claude -p` |

If a `RouteRequest` sets `NeedsTools: true`, the router **refuses** it with
`ErrAgenticNotSupported` and returns no provider. Routing agentic work through
this library is a category error — point it at `claude -p` instead.

---

## The Provider abstraction

```go
type Provider interface {
    Generate(ctx context.Context, prompt, system string) (string, error)
}
```

Two implementations, both constructed with everything **injected** (no global
config, no singleton):

```go
// Local — non-streaming Ollama /api/chat. Zero timeout -> 120s default.
ollama := wrsllm.NewOllamaProvider("http://localhost:11434", "qwen3:8b", 0)

// Cloud — Anthropic Messages API. Zero timeout -> 60s default.
claude := wrsllm.NewClaudeProvider(os.Getenv("ANTHROPIC_API_KEY"), "claude-sonnet-4", 0)
```

The provider logic was lifted verbatim from `openbrain/internal/llm` and
de-coupled from openbrain's `config.Get()` singleton. **`wrs-llm` holds no
package-level state** — every caller constructs its own `RouterConfig`.

---

## The Router

```go
type Router struct{ /* ... */ }

func NewRouter(cfg RouterConfig) *Router
func (r *Router) Route(req RouteRequest) (RouteDecision, error)
```

### Request / decision

```go
type RouteRequest struct {
    TaskKind   TaskKind // PersonaCreation, ConversationalReply, Caption, Summarize, Extract
    Text       string   // reserved for future length/heuristic classification
    Sensitive  bool     // PII/private -> FORCED local, never cloud
    NeedsTools bool      // agentic -> REFUSED (single-shot only)
    Override   string    // explicit model key or provider name; highest precedence after hard guards
    LowConf    bool      // escalation flag; reserved for the future classifier seam
}

type RouteDecision struct {
    Provider Provider
    Model    string
    Reason   string // always populated, for cost-audit logging
}
```

### Layered routing policy (highest precedence first)

1. **Hard guard — agentic refusal.** `NeedsTools == true` → `ErrAgenticNotSupported`, no provider. Beats everything.
2. **Hard guard — sensitive.** `Sensitive == true` → forced to the local provider; `ErrNoLocalProvider` if none is configured. **This beats both `Override` and the kill switch** — PII never leaves the box.
3. **Override.** `Override` matches a configured **model key** (`cfg.Models`) or **provider name** (`cfg.Providers`); otherwise `ErrUnknownOverride`.
4. **Kill switch.** `ForceAllLocal` / `ForceAllClaude` (a global config flag). `ForceAllClaude` still yields to the sensitive hard guard.
5. **Deterministic task-kind map.** `cfg.Routes[TaskKind]` → `{provider, model}`; `ErrNoRoute` if unmapped.
6. **Classifier seam.** Reserved (gated on `LowConf`) — **not built** in this module.

Every decision sets `RouteDecision.Reason` so callers can log *why* a request
went where it did (cost auditing).

### Config — caller-injected

```go
type RouterConfig struct {
    Local      Provider              // required for sensitive routing + ForceAllLocal
    Cloud      Provider              // required for ForceAllClaude + cloud overrides
    Routes     map[TaskKind]Route    // TaskKind -> default {provider, model}
    Providers  map[string]Provider   // provider-name -> Provider (for Override)
    Models     map[string]Route      // model-key -> {provider, model} (for Override)
    KillSwitch KillSwitch            // KillSwitchOff | ForceAllLocal | ForceAllClaude
}
```

The caller builds this from its own environment. A `DefaultRoutes` helper
provides the strategy's default model map:

| TaskKind | Provider | Model |
|---|---|---|
| `Extract` | Ollama (local) | `qwen3:8b` |
| `PersonaCreation` | Ollama (local) | `qwen3:8b` |
| `Summarize` | Ollama (local) | `llama3.1:8b` |
| `ConversationalReply` | Ollama (local) | `llama3.1:8b` |
| `Caption` | Ollama (local) | `llama3.1:8b` |

Claude is **not** in the default map — it is reserved for escalation/override.

### Typed errors

All routing errors are `errors.Is`-matchable:

| Error | Cause |
|---|---|
| `ErrAgenticNotSupported` | `NeedsTools == true` |
| `ErrNoLocalProvider` | sensitive (or force-all-local) with no local provider |
| `ErrNoCloudProvider` | force-all-claude with no cloud provider |
| `ErrNoRoute` | task kind not in `cfg.Routes` |
| `ErrUnknownOverride` | `Override` matches no model key or provider name |

---

## Example

```go
local := wrsllm.NewOllamaProvider("http://localhost:11434", wrsllm.DefaultQwenModel, 0)
cloud := wrsllm.NewClaudeProvider(os.Getenv("ANTHROPIC_API_KEY"), "claude-sonnet-4", 0)

router := wrsllm.NewRouter(wrsllm.RouterConfig{
    Local:  local,
    Cloud:  cloud,
    Routes: wrsllm.DefaultRoutes(local),
    Models: map[string]wrsllm.Route{
        "claude": {Provider: cloud, Model: "claude-sonnet-4"},
    },
    Providers: map[string]wrsllm.Provider{"cloud": cloud},
})

dec, err := router.Route(wrsllm.RouteRequest{TaskKind: wrsllm.Extract})
if err != nil {
    log.Fatal(err)
}
log.Printf("routing: %s", dec.Reason)

out, err := dec.Provider.Generate(ctx, "Extract the dates from: ...", "Return JSON only.")
```

---

## Scope

This is **Phase 0a** of the WRS model-routing initiative: the standalone module
only. It does **not** migrate openbrain (Phase 0b) and wires no consumer. The
module has **no external dependencies** beyond the Go standard library.

## Development

```bash
go test ./... -race   # all tests, race detector
go vet ./...
gofmt -l .            # no output == clean
```
