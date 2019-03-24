package tensor

// Tensor is a performance optimized n-dimensional array
// backed by a one dimensional float32 slice
type Tensor struct {
	data  []float32
	shape Shape
}

// At returns the float32 scalar at index
func (t Tensor) At(index int) float32 {
	return t.data[index]
}

// Count returns the number of top level tensor
func (t Tensor) Count() uint32 {
	return t.Shape()[0]
}

// Data returns all the data in the backing slice
func (t Tensor) Data() []float32 {
	return t.data
}

// Dot calculates the dot between Tensor.data and Tensor b.data
func (t Tensor) Dot(b Tensor) (sum float32) {
	for i := range t.data {
		sum += t.data[i] * b.data[i]
	}

	return
}

// Each takes a function and calls with every top level
// Tensor
func (t Tensor) Each(f func(int, Tensor)) {
	for i := 0; i < int(t.shape[0]); i++ {
		f(i, t.Get(i))
	}
}

// Get returns a top level Tensor at index i
func (t Tensor) Get(i int) Tensor {
	a := i * int(t.shape[1:].Size())
	z := a + int(t.shape[1:].Size())
	return Tensor{data: t.data[a:z], shape: t.shape[1:]}
}

// Len returns the length of slice in the source data
func (t Tensor) Len() uint32 {
	return t.Shape()[len(t.Shape())-1]
}

// SetAt updates a scalar value in the backing float32
// array at the specified coordinates.
func (t Tensor) SetAt(v float32, coordinates ...uint32) {
	var index uint32

	for i := range coordinates {
		if coordinates[i] == 0 {
			continue
		}

		k := uint32(1)

		for j, dimension := range t.shape[i:] {
			if j == 0 {
				dimension = 1
			}

			k = k * coordinates[i] * dimension
		}

		index += k
	}

	t.data[index] = v
}

// Shape returns the dimensions of the data in a tensor
func (t Tensor) Shape() Shape {
	return t.shape
}

// New creates a new Tensor of specified shape
func New(dimensions ...uint32) Tensor {
	t := Tensor{shape: Shape(dimensions)}
	size := t.shape.Size()
	t.data = make([]float32, size, size)
	return t
}
