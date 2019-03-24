package tensor

import (
	"testing"
)

func TestCount(t *testing.T) {
	tests := []struct {
		name  string
		shape []uint32
		count uint32
	}{
		{"2/2/2=2", []uint32{2, 2, 2}, 2},
	}

	for _, test := range tests {
		tensor := New(test.shape...)
		actualCount := tensor.Count()
		if actualCount != test.count {
			t.Errorf("tensor.Count()=%d, expected %d", actualCount, test.count)
		}
	}
}

func TestDot(t *testing.T) {
	tests := []struct {
		name string
		a    []float32
		b    []float32
		sum  float32
	}{
		{"2,3,4.5,6,7", []float32{2.0, 3.0, 4.0}, []float32{5.0, 6.0, 7.0}, 56.0},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := Tensor{data: test.a}
			b := Tensor{data: test.b}
			actualSum := a.Dot(b)
			if actualSum != test.sum {
				t.Errorf("actualSum = %.0f, expected %.0f", actualSum, test.sum)
			}
		})
	}
}

func TestEach(t *testing.T) {
	tests := []struct {
		name  string
		shape []uint32
		data  []float32
	}{
		{"3,3", []uint32{3, 3}, []float32{.01, .02, .03}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tensor := New(3, 3)
			tensor.Each(func(i int, t Tensor) {
				data := t.Data()
				for k, v := range test.data {
					data[k] = v
				}
			})

			for k := range test.data {
				if tensor.data[k] != test.data[k] {
					t.Errorf(
						"tensor.data[%d] = %.2f, expected %.2f",
						k,
						tensor.data[k],
						test.data[k],
					)
				}
			}
		})
	}
}

func TestLen(t *testing.T) {
	tests := []struct {
		name   string
		shape  []uint32
		length uint32
	}{
		{"2/2/2=2", []uint32{2, 2, 2}, 2},
	}

	for _, test := range tests {
		tensor := New(test.shape...)
		actualLength := tensor.Len()
		if actualLength != test.length {
			t.Errorf("tensor.Len()=%d, expected %d", actualLength, test.length)
		}
	}
}

func TestNew(t *testing.T) {
	tests := []struct {
		name  string
		shape []uint32
		size  uint32
	}{
		{"2,3,4==24", []uint32{2, 3, 4}, 24},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tensor := New(test.shape...)
			actualSize := tensor.Shape().Size()
			if actualSize != test.size {
				t.Errorf(
					"tensor.Size() = %d, expected %d",
					actualSize,
					test.size,
				)
			}
		})
	}
}

func TestSetAs(t *testing.T) {
	tests := []struct {
		name        string
		coordinates []uint32
		index       int
	}{
		{"0,0,0=0", []uint32{0, 0, 0}, 0},
		{"1,0,0=0", []uint32{1, 0, 0}, 12},
		{"0,1,0=0", []uint32{0, 1, 0}, 4},
		{"0,0,1=0", []uint32{0, 0, 1}, 1},
	}

	const expectedValue = 0.75
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tensor := New(2, 3, 4)
			tensor.SetAt(expectedValue, test.coordinates...)
			actualValue := tensor.At(test.index)
			if tensor.data[test.index] != expectedValue {
				t.Errorf(
					"tensor.data[%d] = %.2f, expected, %.2f : %v",
					test.index,
					actualValue,
					expectedValue,
					tensor.data,
				)
			}
		})
	}
}
