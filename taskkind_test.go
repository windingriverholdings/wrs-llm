package wrsllm

import "testing"

func TestTaskKind_String(t *testing.T) {
	cases := []struct {
		kind TaskKind
		want string
	}{
		{PersonaCreation, "PersonaCreation"},
		{ConversationalReply, "ConversationalReply"},
		{Caption, "Caption"},
		{Summarize, "Summarize"},
		{Extract, "Extract"},
		{TaskKind(999), "TaskKind(unknown)"},
	}
	for _, c := range cases {
		if got := c.kind.String(); got != c.want {
			t.Errorf("TaskKind(%d).String() = %q, want %q", c.kind, got, c.want)
		}
	}
}
