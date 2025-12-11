package tanh

import (
	"Arion/configs"
	"Arion/pkg/he"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

/*
 * CiphertextMatricesOutputLinear
 * 输入:  ctMatrices *he.CiphertextMatrices
 * 输出: *he.CiphertextMatrices, error
 * 说明: 计算需要得到每个batch的第一个token，因此有许多无用的slots，所以我们可以将一个batch中的多个元素整合到一个密文中计算，
 *       这样在后续进行tanh时，只需要对更少的密文进行tanh拟合即可。
 */

func CiphertextMatricesOutputLinear(ctMatrices *he.CiphertextMatrices, modelParams *configs.ModelParams, ckksParams ckks.Parameters, eval *ckks.Evaluator) (*he.CiphertextMatrices, error) {

	// 返回维数
	ctRows := ctMatrices.NumRow
	ctCols := ctMatrices.NumCol
	ctBatch := ctMatrices.NumBatch
	// fmt.Println("realRow: ", ctRealRows)
	// fmt.Printf("Ciphertext Matrices Batch:%d, Rows:%d, Cols:%d\n", ctBatch, ctRows, ctCols)

	// step1 确定需要多少列数据才能装完所有数据，
	ctNewCols := ctCols / ctRows

	// 进行计算
	newCiphertexts := make([]*rlwe.Ciphertext, ctNewCols)

	biasVec := make([]float64, ctBatch)
	for j := 0; j < ctBatch; j++ {
		biasVec[j] = 1
	}
	for i := 0; i < ctNewCols; i++ {
		ct := ckks.NewCiphertext(ckksParams, ctMatrices.Ciphertexts[0].Degree(), ctMatrices.Ciphertexts[0].Level())
		for j := 0; j < ctRows; j++ {
			// 取出第一个token的值
			ctTmp, err := eval.MulNew(ctMatrices.Ciphertexts[i*ctRows+j], biasVec)
			if err != nil {
				panic(err)
			}

			if err := eval.Rescale(ctTmp, ctTmp); err != nil {
				panic(err)
			}

			// 旋转到指定位置
			ctRot, err := eval.RotateNew(ctTmp, -j*ctBatch)
			if err != nil {
				panic(err)
			}

			ctRot.Scale = ckksParams.DefaultScale()
			// 将结果累加到ct中
			if err := eval.Add(ct, ctRot, ct); err != nil {
				panic(err)
			}
		}
		newCiphertexts[i] = ct
	}

	// 返回结果
	return &he.CiphertextMatrices{
		Ciphertexts: newCiphertexts,
		NumBatch:    ctMatrices.NumBatch,
		NumRow:      ctRows,
		NumCol:      ctNewCols,
	}, nil
}
