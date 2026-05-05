// Package logger provides a lightweight scalar/histogram metrics sink
// for stlcg training runs, plus a gomlx train.Trainer-compatible
// RobustnessMetric.
//
// The v1 sink is stdlib-only JSONL; each call to Scalar or Histogram
// writes one JSON record per line. TensorBoard summaries are deferred —
// users who want them can pipe JSONL through a converter.
package logger

import (
	"encoding/json"
	"io"
	"sync"
	"time"
)

// Logger is the minimal scalar/histogram sink contract.
//
// Implementations must be safe for concurrent use; the caller makes no
// ordering guarantees across goroutines.
type Logger interface {
	Scalar(tag string, step int64, value float64)
	Histogram(tag string, step int64, values []float64)
	Close() error
}

// NopLogger discards all records. Safe for concurrent use.
type NopLogger struct{}

func (NopLogger) Scalar(string, int64, float64)         {}
func (NopLogger) Histogram(string, int64, []float64)    {}
func (NopLogger) Close() error                          { return nil }

// JSONLLogger writes one JSON record per line to an io.Writer.
//
// Record shape:
//
//	{"kind":"scalar","tag":"loss","step":42,"value":0.123,"ts":"2026-05-05T12:34:56Z"}
//	{"kind":"hist","tag":"grad","step":42,"values":[...],"ts":"..."}
//
// Close flushes if the underlying writer is an io.Closer.
type JSONLLogger struct {
	mu     sync.Mutex
	w      io.Writer
	enc    *json.Encoder
	closed bool
}

// NewJSONL wraps w as a JSONLLogger. The caller retains ownership of w;
// Close is a no-op unless w implements io.Closer, in which case w.Close
// is called.
func NewJSONL(w io.Writer) *JSONLLogger {
	return &JSONLLogger{w: w, enc: json.NewEncoder(w)}
}

type scalarRecord struct {
	Kind  string    `json:"kind"`
	Tag   string    `json:"tag"`
	Step  int64     `json:"step"`
	Value float64   `json:"value"`
	TS    time.Time `json:"ts"`
}

type histogramRecord struct {
	Kind   string    `json:"kind"`
	Tag    string    `json:"tag"`
	Step   int64     `json:"step"`
	Values []float64 `json:"values"`
	TS     time.Time `json:"ts"`
}

func (l *JSONLLogger) Scalar(tag string, step int64, value float64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return
	}
	_ = l.enc.Encode(scalarRecord{Kind: "scalar", Tag: tag, Step: step, Value: value, TS: time.Now().UTC()})
}

func (l *JSONLLogger) Histogram(tag string, step int64, values []float64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return
	}
	cp := make([]float64, len(values))
	copy(cp, values)
	_ = l.enc.Encode(histogramRecord{Kind: "hist", Tag: tag, Step: step, Values: cp, TS: time.Now().UTC()})
}

func (l *JSONLLogger) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return nil
	}
	l.closed = true
	if c, ok := l.w.(io.Closer); ok {
		return c.Close()
	}
	return nil
}
