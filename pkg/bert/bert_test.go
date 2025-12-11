package bert

import (
	"Arion/configs"
	"Arion/pkg/ecdmat"
	"Arion/pkg/he"
	myutils "Arion/pkg/utils"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils"
	"gonum.org/v1/gonum/mat"
)

func TestComputeAttention(t *testing.T) {
	// 使用小规模模型参数
	modelParams := &configs.ModelParams{
		NumBatch:  2,
		NumRow:    8,
		NumCol:    4,
		SqrtD:     2.0,
		BabyStep:  3,
		GiantStep: 3,
		InvIter:   2,
	}

	// 初始化简单的CKKS参数
	ckksParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            5,
		LogQ:            []int{58, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
		LogP:            []int{60, 60, 60},
		LogDefaultScale: 46,
	})
	if err != nil {
		t.Fatalf("CKKS参数初始化失败: %v", err)
	}
	// 2. Build bootstrapping circuit parameters
	btpLit := bootstrapping.ParametersLiteral{
		LogN: utils.Pointy(ckksParams.LogN()), // match residual LogN
		LogP: []int{60, 60, 60},               // must match ckksParams.LogP
		// LogP: []int{55, 55, 55, 55},
		Xs: ckksParams.Xs(), // same secret distribution
	}
	btpParams, err := bootstrapping.NewParametersFromLiteral(ckksParams, btpLit)
	if err != nil {
		panic(err)
	}

	// 随机生成输入矩阵 inputQ, inputK, inputV，范围[-1,1)
	inputQ := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
	inputK := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
	inputV := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
	for i := 0; i < modelParams.NumRow; i++ {
		for j := 0; j < modelParams.NumCol; j++ {
			inputQ.Set(i, j, rand.Float64()*2-1)
			inputK.Set(i, j, rand.Float64()*2-1)
			inputV.Set(i, j, rand.Float64()*2-1)
		}
	}
	// 编码、批处理
	inputQ = myutils.PadOrTruncateMatrix(inputQ, modelParams)
	inputK = myutils.PadOrTruncateMatrix(inputK, modelParams)
	inputV = myutils.PadOrTruncateMatrix(inputV, modelParams)
	inputMatsQ := myutils.MatrixToBatchMats(inputQ, modelParams)
	inputMatsK := myutils.MatrixToBatchMats(inputK, modelParams)
	inputMatsV := myutils.MatrixToBatchMats(inputV, modelParams)
	encodedInputsQ := ecdmat.EncodeDense(inputMatsQ, modelParams)
	encodedInputsK := ecdmat.EncodeDense(inputMatsK, modelParams)
	encodedInputsV := ecdmat.EncodeDense(inputMatsV, modelParams)

	// 生成密钥
	ecd, enc, eval, dec, btpEval := he.GenerateKeysAndBtsKeys(ckksParams, btpParams, modelParams)

	// 加密
	ctQ, err := he.EncryptInputMatrices(encodedInputsQ, modelParams, ckksParams, enc, ecd)
	if err != nil {
		t.Fatalf("加密Q失败: %v", err)
	}
	ctK, err := he.EncryptInputMatrices(encodedInputsK, modelParams, ckksParams, enc, ecd)
	if err != nil {
		t.Fatalf("加密K失败: %v", err)
	}
	ctV, err := he.EncryptInputMatrices(encodedInputsV, modelParams, ckksParams, enc, ecd)
	if err != nil {
		t.Fatalf("加密V失败: %v", err)
	}

	// 调用Halevi-Shoup密文矩阵attention主流程
	fmt.Println("开始计算attention时密文层级：", ctQ.Ciphertexts[0].Level())
	ctExpQKTMulV, ctSumInv, err := ComputeAttention(ctQ, ctK, ctV, modelParams, &ckksParams, ecd, enc, eval, btpEval)
	if err != nil {
		t.Fatalf("Halevi-Shoup密文矩阵attention计算失败: %v", err)
	}
	fmt.Println("结束计算attention时密文层级：", ctExpQKTMulV.Ciphertexts[0].Level())

	// 解密
	decY, err := he.DecryptCiphertextMatrices(ctExpQKTMulV, modelParams, ckksParams, dec, ecd)
	if err != nil {
		t.Fatalf("解密失败: %v", err)
	}

	// 解码
	matsY := ecdmat.DecodeDense(decY, modelParams)

	// 输出输入矩阵 inputQ
	t.Log("输入矩阵Q（inputQ）：")
	myutils.PrintMat(inputQ)
	// 输出输入矩阵 inputK
	t.Log("输入矩阵K（inputK）：")
	myutils.PrintMat(inputK)
	// 输出输入矩阵 inputV
	t.Log("输入矩阵V（inputV）：")
	myutils.PrintMat(inputV)

	// 明文直接计算 Q × K^T
	expectedQK := mat.NewDense(modelParams.NumRow, modelParams.NumRow, nil)
	kT := mat.DenseCopyOf(inputK.T())
	expectedQK.Product(inputQ, kT)

	// 对 expectedQK 做 softmax（每行）
	softmaxQK := mat.NewDense(modelParams.NumRow, modelParams.NumRow, nil)
	expQK := mat.NewDense(modelParams.NumRow, modelParams.NumRow, nil)
	var sumExpQKbyRow []float64
	for i := 0; i < modelParams.NumRow; i++ {

		// 计算exp并累加
		sum := 0.0
		for j := 0; j < modelParams.NumRow; j++ {
			v := math.Exp(expectedQK.At(i, j))
			expQK.Set(i, j, v)
			sum += v
		}
		// 归一化
		for j := 0; j < modelParams.NumRow; j++ {
			softmaxQK.Set(i, j, expQK.At(i, j)/sum)
		}
		sumExpQKbyRow = append(sumExpQKbyRow, sum)
	}
	// 明文 Exp(QK^T)×V
	expQKV := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
	expQKV.Product(expQK, inputV)

	// 对 expQKV 做 softmax（每行）
	expQKVDivSum := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
	for i := 0; i < modelParams.NumRow; i++ {
		for j := 0; j < modelParams.NumCol; j++ {
			expQKVDivSum.Set(i, j, expQKV.At(i, j)/sumExpQKbyRow[i])
		}
	}
	var sumExpQKInv []float64
	for i := 0; i < modelParams.NumRow; i++ {
		sumExpQKInv = append(sumExpQKInv, 1/sumExpQKbyRow[i])
	}

	// 明文 attention: softmax(QK^T) × V
	expectedAtt := mat.NewDense(modelParams.NumRow, modelParams.NumCol, nil)
	expectedAtt.Product(softmaxQK, inputV)

	// 输出明文 QK^T
	t.Log("明文 QK^T 结果（expectedQK）：")
	myutils.PrintMat(expectedQK)

	t.Log("明文 Exp(QK^T-Max) 结果：")
	myutils.PrintMat(expQK)

	t.Log("明文 1/Sum(Exp(QK^T-Max)) 结果：")
	fmt.Println(sumExpQKInv)

	t.Log("明文 Exp(QK^T-Max)×V 结果：")
	myutils.PrintMat(expQKV)

	t.Log("明文 Exp(QK^T)×V/Sum(Exp(QK^T)) 结果：")
	myutils.PrintMat(expQKVDivSum)

	t.Log("期望 attention Softmax(QK^T)×V 结果：")
	myutils.PrintMat(expectedAtt)

	// 检查第一个batch的结果与明文计算是否接近
	got := matsY[0]
	t.Log("密文 attention 结果（Softmax(QK^T)×V）：")
	myutils.PrintMat(got)

	t.Log("密文1/Sum(Exp(Q×K^T))：")
	ptSumInv := he.DecryptCiphertext(ctSumInv, ckksParams, dec, ecd)
	fmt.Println(ptSumInv)
}
