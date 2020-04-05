package tensor

// Shape defines the dimensions of data stored in a Tensor
type Shape []uint32

// Size returns the scalar capacity of a Tensor of its Shape
func (s Shape) Size() (size uint32) {
	size = 1

	for _, n := range s {
		size = size * n
	}

	return
}
