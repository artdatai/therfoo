package losses

import (
	"github.com/therfoo/therfoo/tensor"
	"math"
)

func CrossEntropy(yEstimate, yTrue []float32) (sum float32) {
	for i := range yTrue {
		sum += -yTrue[i]*float32(math.Log(float64(yEstimate[i]))) - (1-yTrue[i])*float32(math.Log(float64(1-yEstimate[i])))
	}
	return
}

func CrossEntropyPrime(yEstimate, yTrue, delta []float32) {
	for i := 0; i < len(yTrue); i++ {
		delta[i] += yEstimate[i] - yTrue[i]
	}
}
