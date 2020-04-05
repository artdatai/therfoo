package activation

import (
	"math"
)

func Sigmoid(z, a []float32) {
	for i := range z {
		a[i] = 1. / (1. + float32(math.Exp(float64(-z[i]))))
	}
}

func SigmoidDelta(a, aDelta []float32) {
	for i := range a {
		aDelta[i] = a[i] * (1. - a[i])
	}
}
