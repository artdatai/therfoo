package activation

func ReLU(z, a []float32) {
	for i := range z {
		if z[i] < 0. {
			a[i] = 0.
		} else {
			a[i] = z[i]
		}
	}
}

func ReLUDelta(a, aDelta []float32) {
	for i := range a {
		if a[i] > 0. {
			aDelta[i] = 1.
		} else {
			aDelta[i] = 0
		}
	}
}
