package ecdmat

import (
	"Arion/configs"

	"gonum.org/v1/gonum/mat"
)

// EncodeDense 将 numBatch 个 [NumRow×NumCol] 的 *mat.Dense，按列优先方式编码为 NumCol 个长度为 numBatch*NumRow 的 float64 向量
// 遍历顺序：先遍历 row，再遍历 batch
func EncodeDense(mats []*mat.Dense, params *configs.ModelParams) [][]float64 {
	numBatch := params.NumBatch
	if numBatch != len(mats) {
		panic("number of matrices does not match numBatch")
	}
	rows, cols := mats[0].Dims()

	result := make([][]float64, cols)
	for c := 0; c < cols; c++ {
		vec := make([]float64, numBatch*rows)
		for r := 0; r < rows; r++ {
			for b := 0; b < numBatch; b++ {
				vec[r*numBatch+b] = mats[b].At(r, c)
			}
		}
		result[c] = vec
	}
	return result
}

// DecodeDense 将 cols 个长度为 numBatch*rows 的 float64 向量，解码为 numBatch 个 [rows×cols] 的 *mat.Dense
// 遍历顺序：先遍历 row，再遍历 batch
func DecodeDense(vecs [][]float64, params *configs.ModelParams) []*mat.Dense {
	cols := len(vecs)
	if cols == 0 || params.NumBatch == 0 {
		return nil
	}
	numBatch := params.NumBatch
	rows := params.NumRow

	result := make([]*mat.Dense, numBatch)
	for b := 0; b < numBatch; b++ {
		data := make([]float64, rows*cols)
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				data[r*cols+c] = vecs[c][r*numBatch+b]
			}
		}
		result[b] = mat.NewDense(rows, cols, data)
	}
	return result
}

// 由于pooling将输入的每个token都映射到一个新的token上，因此我们可以将一个batch中的多个元素整合到一个密文中计算，
// 因此这个解密是单独为pooling设计的，返回的结果是一个batch中的多个token
func DecodeOutputDenseToMatrix(vecs [][]float64, params *configs.ModelParams) *mat.Dense {
	numBatch := params.NumBatch
	rows := params.NumRow
	cols := len(vecs)
	if cols == 0 || numBatch == 0 {
		return nil
	}
	data := make([]float64, numBatch*cols*rows)
	for b := 0; b < numBatch; b++ {
		for c := 0; c < cols; c++ {
			for d := 0; d < rows; d++ {
				data[b*cols*rows+c*rows+d] = vecs[c][d*numBatch+b]
			}
		}
	}
	return mat.NewDense(numBatch, cols*rows, data)
}
