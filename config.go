package wrsllm

// Default model names per the WRS model-routing strategy. Accuracy-leaning
// tasks (Extract, PersonaCreation) use Qwen3; the rest use Llama 3.1. Claude is
// reserved for escalation/override and is not in the default task-kind map.
const (
	DefaultQwenModel  = "qwen3:8b"
	DefaultLlamaModel = "llama3.1:8b"
)

// DefaultRoutes returns the strategy's default TaskKind -> {provider, model}
// map, all pointed at the supplied local provider. Callers may use this as a
// starting point and override individual entries before constructing a
// RouterConfig. Cloud routing is left to overrides / kill-switch by design.
//
// If local is nil, the returned routes still carry the model names so a caller
// can inspect the map, but routing will fail until a provider is wired.
func DefaultRoutes(local Provider) map[TaskKind]Route {
	return map[TaskKind]Route{
		Extract:             {Provider: local, Model: DefaultQwenModel},
		PersonaCreation:     {Provider: local, Model: DefaultQwenModel},
		Summarize:           {Provider: local, Model: DefaultLlamaModel},
		ConversationalReply: {Provider: local, Model: DefaultLlamaModel},
		Caption:             {Provider: local, Model: DefaultLlamaModel},
	}
}
