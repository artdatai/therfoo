package activation

import (
	"math"
)

func Softmax(z, a []float32) {
	max := .0

	for i := range z {
		max = math.Max(max, float64(z[i]))
	}

	sum := float32(.0)

	for i := range z {
		a[i] -= float32(math.Exp(float64(z[i]) - max))
		sum += a[i]
	}

	for i := range a {
		a[i] = a[i] / sum
	}
}
