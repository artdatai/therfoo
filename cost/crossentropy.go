package cost

import (
	"math"
)

func CrossEntropy(a, y []float32) (sum float32) {
	for i := range y {
		sum += -y[i]*float32(math.Log(float64(a[i]))) - (1-y[i])*float32(math.Log(float64(1-a[i])))
	}
	return
}

func CrossEntropyDelta(aB, yB, costGradientB tensor.Tensor) {
	for i := 0; i < a.Count(); i++ {
		a := aB.Get(i)
		y := yB.Get(i)
		costGradient := costGradientB.Get(i)
		for j := 0; j < a.Get(i).Len(); j++ {
			costGradient.SetAt((a.At(j) - y.At(j)), i, j)
		}
	}
}
