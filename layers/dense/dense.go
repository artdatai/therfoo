package dense

import (
	//	"bytes"
	//	"encoding/gob"
	. "github.com/therfoo/therfoo/tensor"
	//	"math/rand"
)

// Dense is a densely connected neural network layer
type Dense struct {
	activate     func(z, a []float32)
	derive       func(a, aDelta []float32)
	aBatch       Tensor
	aDeltaBatch  Tensor
	biasNabla    Tensor
	weightNabla  Tensor
	zBatch       Tensor
	biasesBatch  Tensor
	weightsBatch Tensor
}

// Activate takes a sample batch and returns the activated weights Dot product
func (d *Dense) Activate(xBatch Tensor) (aBatch Tensor) {
	var biases, x, weights Tensor
	var i, j uint32
	var z float32
	// for example in batch
	for i = 0; i < xBatch.Count(); i++ {
		// x is one example
		x = xBatch.Get(i)
		// w is all corresponding weights
		weights = d.weightsBatch.Get(i)
		// b is all the correspoding biases
		biases = d.biasesBatch.Get(i)
		// for each neuron in corresponding neurons
		for j = 0; j < weights.Count(); j++ {
			// calculate dot(x, weights) product and add a bias
			z = x.Dot(weights.Get(j)) + biases.Get(j).At(0)
			// store z for use in backpropagation
			d.zBatch.SetAt(z, i, j)
		}
		// activate corresponding d.zBatch and store the activation in d.aBatch
		d.activate(d.zBatch.Get(i).Data(), d.aBatch.Get(i).Data())
	}

	return d.aBatch
}

// Minimize takes the batched cost gradient from the next layer, calculates
// the rate of change of the cost with changes to zBatch
func (d *Dense) Minimize(costGradientB Tensor, learningRate, regularization float32) Tensor {
	// a.Delta is set to nil when using cross entropy cost function
	if d.derive != nil {
		d.derive(d.aBatch, d.aDeltaBatch)
		costGradientB.Schur(d.aDeltaBatch)
	}

	// reset bias adjustments
	for i := 0; i < d.biasNabla.Len(); i++ {
		d.biasNabla.SetAt(0, i)
	}

	// sum up bias cost gradients in each batch
	for i := 0; i < costGradientB.Count(); i++ {
		//for each neuron
		for j := 0; j < costGradientB.Get(i).Len(); j++ {
			d.biasNabla.SetAt(
				d.biasNabla.At(i)+costGradient.Get(i).At(j),
				j,
			)
		}
	}

	learningRate = learningRate / costGradient.Count()

	// adjust biases
	for i := 0; i < d.biasesBatch.Count(); i++ {
		for j := 0; j < d.biasesBatch.Get(i).Len(); j++ {
			d.biasesBatch.SetAt(
				d.biasesBatch.Get(i).At(j)-learningRate*d.biasNabla.At(j),
				i,
				j,
			)
		}
	}

	// calculate previousCostGradient before adjusting weight
	// d.previousCostGradient * weight

	// adjust weights
	// nablaWeight = costGradient * x
	// weights = regularization*weights - learningRate*costGradient*nablaWeight

	return d.previousCostGradient
}

//  func (d *Dense) FeedForward(x, z, a tensor.Tensor) {
//  	d.neurons.Each(func(i int, weights Tensor) {
//  		z.SetAt(
//  			x.Dot(weights)+d.Biases.Get(i).At(0),
//  			i,
//  		)
//  		z.Apply(d.activate, a)
//  	})
//  }
//
//  func (d *Dense) Activate(x, a tensor.Tensor) {
//  	z.Apply(d.activate, a)
//  	z := make(tensor.Vector, d.neuronsCount, d.neuronsCount)
//  	for neuron := range d.weights {
//  		sum := 0.
//  		for weight := range d.weights[neuron] {
//  			if weight == 0 {
//  				sum += d.weights[neuron][weight]
//  			} else {
//  				sum += d.weights[neuron][weight] * (*x)[weight-1]
//  			}
//  		}
//  		z[neuron] = sum
//  	}
//
//  	return d.activate(&z)
//  }
//
//  func (d *Dense) Adjust(delta *[][]float64) {
//  	for n := range *delta {
//  		for p := range (*delta)[n] {
//  			d.weights[n][p] -= (*delta)[n][p]
//  		}
//  	}
//  }
//
//  func (d *Dense) Bytes() (weights []byte, err error) {
//  	var b bytes.Buffer
//  	err = gob.NewEncoder(&b).Encode(d.weights)
//  	if err == nil {
//  		weights = b.Bytes()
//  	}
//  	return
//  }
//
//  func (d *Dense) Derive(activation, cost *tensor.Vector) {
//  	d.derive(activation).Each(func(index int, value float64) {
//  		(*cost)[index] = (*cost)[index] * value
//  	})
//  }
//
//  func (d *Dense) Init(neuronsCount, weightsCount int) {
//  	d.neuronsCount = neuronsCount
//  	d.weightsCount = weightsCount
//  	d.weights = make([][]float64, d.neuronsCount, d.neuronsCount)
//  	totalWeights := d.weightsCount + 1
//  	for n := range d.weights {
//  		d.weights[n] = make([]float64, totalWeights, totalWeights)
//  		for w := range d.weights[n] {
//  			d.weights[n][w] = rand.Float64()
//  		}
//  	}
//  }
//
//  func (d *Dense) Load(b []byte) error {
//  	return gob.NewDecoder(bytes.NewBuffer(b)).Decode(&d.weights)
//  }
//
//  func (d *Dense) NextCost(cost *tensor.Vector) *tensor.Vector {
//  	next := make(tensor.Vector, d.weightsCount, d.weightsCount)
//  	for n := range d.weights {
//  		for w := range d.weights[n] {
//  			if w > 0 {
//  				next[w-1] += d.weights[n][w-1] * (*cost)[n]
//  			}
//  		}
//  	}
//  	return &next
//  }
//
//  func (d *Dense) Size() int {
//  	return d.neuronsCount
//  }
//
//  func New(options ...Option) *Dense {
//  	d := Dense{}
//  	for i := range options {
//  		options[i](&d)
//  	}
//  	return &d
//  }
