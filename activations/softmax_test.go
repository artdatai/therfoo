package activations

import (
	"fmt"
	"testing"
)

func TestSoftmax(t *testing.T) {
	tests := []struct {
		name     string
		z        []float32
		expected []string
	}{
		{"1,2,3", []float32{1., 2., 3.}, []string{"0.0900", "0.2447", "0.6652"}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := []float32{0, 0, 0}
			Softmax(test.z, a)
			for i := range test.expected {
				actual := fmt.Sprintf("%.4f", a[i])
				expected := test.expected[i]
				if actual != expected {
					t.Errorf("%.0f = %s, expected %s", test.z[i], actual, expected)
				}
			}
		})
	}
}
