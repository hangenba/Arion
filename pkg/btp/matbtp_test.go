package btp

import (
	"Arion/configs"
	"Arion/pkg/ecdmat"
	"Arion/pkg/he"
	myutils "Arion/pkg/utils" // 本地 utils 包，起别名 myutils
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils"
	"gonum.org/v1/gonum/mat"
)

func TestCiphertextMatricesBootstrapping(t *testing.T) {
	// 初始化简单的CKKS参数和bootstrapping参数
	LogN := 10
	ckksParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            LogN,
		LogQ:            []int{56, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45},
		LogP:            []int{55, 55, 55, 55},
		LogDefaultScale: 45,
	})
	if err != nil {
		t.Fatalf("CKKS参数初始化失败: %v", err)
	}
	modelParams := &configs.ModelParams{
		NumBatch: 256,
		NumRow:   2,
		NumCol:   4,
		SqrtD:    1.0,
	}

	// 生成密钥和bootstrapping evaluator
	btpParams, err := bootstrapping.NewParametersFromLiteral(ckksParams, bootstrapping.ParametersLiteral{
		LogN: utils.Pointy(LogN),
		LogP: []int{60, 60, 60},
		Xs:   ckksParams.Xe(),
	})
	if err != nil {
		t.Fatalf("Bootstrapping参数初始化失败: %v", err)
	}
	ecd, enc, eval, dec, btpEval := he.GenerateKeysAndBtsKeys(ckksParams, btpParams, modelParams)

	// 构造明文矩阵
	values := make([]float64, modelParams.NumRow*modelParams.NumCol)
	for i := range values {
		values[i] = rand.Float64() // 或者用你需要的填充值
	}
	matData := mat.NewDense(modelParams.NumRow, modelParams.NumCol, values)
	batchMats := myutils.MatrixToBatchMats(matData, modelParams)
	encoded := ecdmat.EncodeDense(batchMats, modelParams)

	_ = eval
	// 加密
	ctMats, err := he.EncryptInputMatrices(encoded, modelParams, ckksParams, enc, ecd)
	if err != nil {
		t.Fatalf("加密失败: %v", err)
	}
	for i := 0; i < ctMats.NumCol; i++ {
		eval.DropLevel(ctMats.Ciphertexts[i], 10) // 降低密文层级
	}
	fmt.Println("Bootstrapping之前的密文层级", ctMats.Ciphertexts[0].Level())
	fmt.Println("Bootstrapping之前的Scale", ctMats.Ciphertexts[0].Scale)

	ctMatsAnother, err := he.EncryptInputMatrices(encoded, modelParams, ckksParams, enc, ecd)
	if err != nil {
		t.Fatalf("加密失败: %v", err)
	}

	// 执行bootstrapping
	ctMatsNew, err := CiphertextMatricesBootstrapping(ctMats, btpEval)
	if err != nil {
		t.Fatalf("Bootstrapping失败: %v", err)
	}
	fmt.Println("Bootstrapping之后的密文层级", ctMatsNew.Ciphertexts[0].Level())
	fmt.Println("Bootstrapping之后的Scale", ctMatsNew.Ciphertexts[0].Scale)

	// 解密
	decRes, err := he.DecryptCiphertextMatrices(ctMatsNew, modelParams, ckksParams, dec, ecd)
	if err != nil {
		t.Fatalf("解密失败: %v", err)
	}

	// eval.DropLevel(ctMatsNew.Ciphertexts[0], ctMatsNew.Ciphertexts[0].Level()-ctMats.Ciphertexts[0].Level())
	eval.MulNew(ctMatsAnother.Ciphertexts[0], ctMatsNew.Ciphertexts[0]) // 乘法测试

	// 输出结果
	t.Log("Bootstrapping后解密结果：")
	for i, row := range decRes {
		t.Logf("Row %d: %v", i, row)
	}
}

func TestCiphertextMatricesBootstrappingMT(t *testing.T) {
	// 初始化简单的CKKS参数和bootstrapping参数
	LogN := 10
	ckksParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            LogN,
		LogQ:            []int{56, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45},
		LogP:            []int{55, 55, 55, 55},
		LogDefaultScale: 45,
	})
	if err != nil {
		t.Fatalf("CKKS参数初始化失败: %v", err)
	}
	modelParams := &configs.ModelParams{
		NumBatch: 256,
		NumRow:   2,
		NumCol:   768,
		SqrtD:    1.0,
	}

	// 生成密钥和bootstrapping evaluator
	btpParams, err := bootstrapping.NewParametersFromLiteral(ckksParams, bootstrapping.ParametersLiteral{
		LogN: utils.Pointy(LogN),
		LogP: []int{60, 60, 60},
		Xs:   ckksParams.Xe(),
	})
	if err != nil {
		t.Fatalf("Bootstrapping参数初始化失败: %v", err)
	}
	ecd, enc, eval, dec, btpEval := he.GenerateKeysAndBtsKeys(ckksParams, btpParams, modelParams)

	// 构造明文矩阵
	values := make([]float64, modelParams.NumRow*modelParams.NumCol)
	for i := range values {
		values[i] = rand.Float64() // 或者用你需要的填充值
	}
	matData := mat.NewDense(modelParams.NumRow, modelParams.NumCol, values)
	batchMats := myutils.MatrixToBatchMats(matData, modelParams)
	encoded := ecdmat.EncodeDense(batchMats, modelParams)

	_ = eval
	// 加密
	ctMats, err := he.EncryptInputMatrices(encoded, modelParams, ckksParams, enc, ecd)
	if err != nil {
		t.Fatalf("加密失败: %v", err)
	}
	for i := 0; i < ctMats.NumCol; i++ {
		eval.DropLevel(ctMats.Ciphertexts[i], 14) // 降低密文层级
	}
	fmt.Println("Bootstrapping之前的密文层级", ctMats.Ciphertexts[0].Level())
	fmt.Println("Bootstrapping之前的Scale", ctMats.Ciphertexts[0].Scale)

	// 多线程执行bootstrapping
	numThreads := 12
	ctMatsNew, err := CiphertextMatricesBootstrappingMT(ctMats, btpEval, btpParams, numThreads)
	if err != nil {
		t.Fatalf("BootstrappingMT失败: %v", err)
	}
	fmt.Println("BootstrappingMT之后的密文层级", ctMatsNew.Ciphertexts[0].Level())
	fmt.Println("BootstrappingMT之后的Scale", ctMatsNew.Ciphertexts[0].Scale)

	// 解密
	decRes, err := he.DecryptCiphertextMatrices(ctMatsNew, modelParams, ckksParams, dec, ecd)
	if err != nil {
		t.Fatalf("解密失败: %v", err)
	}

	t.Log("BootstrappingMT后解密结果：")
	for i, row := range decRes {
		t.Logf("Row %d: %v", i, row)
	}
}
