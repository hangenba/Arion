package he

import (
	"Arion/configs"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// DecryptCiphertextMatrices 对 CiphertextMatrices 进行解密，返回 [][]float64，每个元素为解密后的明文向量
func DecryptCiphertextMatrices(ctMats *CiphertextMatrices, modelParams *configs.ModelParams, ckksParams ckks.Parameters, dec *rlwe.Decryptor, ecd *ckks.Encoder) ([][]float64, error) {
	num := len(ctMats.Ciphertexts)
	result := make([][]float64, num)
	for i, ct := range ctMats.Ciphertexts {
		value := make([]float64, modelParams.NumBatch*modelParams.NumRow)
		// 进行解密操作
		pt := dec.DecryptNew(ct)

		// 将解密后的明文转换为 []float64
		if err := ecd.Decode(pt, value); err != nil {
			panic(err)
		}
		result[i] = value
	}
	return result, nil
}

func DecryptCiphertext(ct *rlwe.Ciphertext, ckksParams ckks.Parameters, dec *rlwe.Decryptor, ecd *ckks.Encoder) []float64 {
	value := make([]float64, ckksParams.MaxSlots())
	pt := dec.DecryptNew(ct)
	if err := ecd.Decode(pt, value); err != nil {
		panic(err)
	}
	return value
}
