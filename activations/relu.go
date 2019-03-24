package activations

func ReLU(z, a []float32) {
	for i := range z {
		if z[i] < 0. {
			a[i] = 0.
		} else {
			a[i] = z[i]
		}
	}
}

func ReLUPrime(a, aPrime []float32) {
	for i := range a {
		if a[i] > 0. {
			aPrime[i] = 1.
		} else {
			aPrime[i] = 0
		}
	}
}
