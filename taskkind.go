package wrsllm

// TaskKind identifies the kind of single-shot work a request represents. The
// router maps each TaskKind to a configured {provider, model} pair.
type TaskKind int

const (
	// PersonaCreation builds a persona/profile from input — accuracy-leaning.
	PersonaCreation TaskKind = iota
	// ConversationalReply generates a chat-style reply.
	ConversationalReply
	// Caption produces a short caption for media.
	Caption
	// Summarize condenses input text.
	Summarize
	// Extract pulls structured facts from text — accuracy-leaning.
	Extract
)

// String renders a TaskKind for logging and Reason strings.
func (k TaskKind) String() string {
	switch k {
	case PersonaCreation:
		return "PersonaCreation"
	case ConversationalReply:
		return "ConversationalReply"
	case Caption:
		return "Caption"
	case Summarize:
		return "Summarize"
	case Extract:
		return "Extract"
	default:
		return "TaskKind(unknown)"
	}
}
