package he

import (
	"Arion/configs"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type CiphertextMatrices struct {
	Ciphertexts []*rlwe.Ciphertext
	NumBatch    int
	NumRow      int
	NumCol      int
}

// he/ciphertext_matrices.go
func (m *CiphertextMatrices) CopyNew() *CiphertextMatrices {
	out := &CiphertextMatrices{
		Ciphertexts: make([]*rlwe.Ciphertext, len(m.Ciphertexts)),
		NumBatch:    m.NumBatch, NumRow: m.NumRow, NumCol: m.NumCol,
	}
	for i, ct := range m.Ciphertexts {
		if ct != nil {
			out.Ciphertexts[i] = ct.CopyNew() // ★ 关键：库内深拷贝，含Scale/Meta
		}
	}
	return out
}

func EncryptInputMatrices(inputMatrices [][]float64, modelParams *configs.ModelParams, ckksParams ckks.Parameters, enc *rlwe.Encryptor, ecd *ckks.Encoder) (*CiphertextMatrices, error) {
	ciphertexts := make([]*rlwe.Ciphertext, len(inputMatrices))
	pt := ckks.NewPlaintext(ckksParams, ckksParams.MaxLevel())
	for i, mat := range inputMatrices {
		// Encrypt each matrix row-wise or column-wise as needed

		if err := ecd.Encode(mat, pt); err != nil {
			panic(err)
		}

		ct, err := enc.EncryptNew(pt)
		if err != nil {
			return nil, err
		}
		ciphertexts[i] = ct
	}
	return &CiphertextMatrices{
		Ciphertexts: ciphertexts,
		NumBatch:    modelParams.NumBatch,
		NumRow:      modelParams.NumRow,
		NumCol:      len(inputMatrices),
	}, nil
}
