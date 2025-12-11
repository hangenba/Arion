package configs

import (
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils"
)

type ModelParams struct {
	NumBatch    int       // 对应 num_batch
	NumRow      int       // 对应 num_row
	NumCol      int       // 对应 num_col
	NumInter    int       // 对应 num_inter
	NumRealRow  int       // 对应 Real compute num_row
	SqrtD       float64   // 对应 sqrt_d
	NumLayers   int       // 对应 num_layers
	NumHeads    int       // 对应 num_heads
	BabyStep    int       // 对应 baby_step
	GiantStep   int       // 对应 giant_step
	ExpSubValue []float64 // 计算Eep(x-ExpSubValue)
	ModelPath   string    // 模型文件路径

	// 近似Exp的参数
	ExpMinValue float64
	ExpMaxValue float64
	ExpDegree   int

	ConstantValue float64 // 计算softmax时的常数系数

	// 近似1/x的参数
	InvMinValue float64
	InvMaxValue float64
	InvDegree   int
	InvIter     int

	// layernorm1近似1/sqrt(x)的参数
	InvSqrtMinValue1 float64
	InvSqrtMaxValue1 float64
	InvSqrtDegree1   int
	InvSqrtIter1     int

	// 近似GELU的参数
	GeluMinValue float64
	GeluMaxValue float64
	GeluDegree   int

	// layernorm2近似1/sqrt(x)的参数
	InvSqrtMinValue2 float64
	InvSqrtMaxValue2 float64
	InvSqrtDegree2   int
	InvSqrtIter2     int

	// Llama paramter
	NumColKV int
	GroupQ   int
	GroupKV  int
}

func InitLlama(paramType string, rowValue int, modelPath string) (ckksParams ckks.Parameters, btpParams bootstrapping.Parameters, modelParams *ModelParams, err error) {

	var setParamsFunc func() (ckks.Parameters, bootstrapping.Parameters, error)
	var batchSize int
	switch paramType {
	case "tiny":
		setParamsFunc = SetHEParamsTiny
	case "short":
		setParamsFunc = SetHEParamsShort
	case "base":
		setParamsFunc = SetHEParams
	default:
		return ckks.Parameters{}, bootstrapping.Parameters{}, nil, fmt.Errorf("不支持的参数类型: %s", paramType)
	}

	ckksParams, btpParams, err = setParamsFunc()
	if err != nil {
		return ckks.Parameters{}, bootstrapping.Parameters{}, nil, fmt.Errorf("failed to set HE parameters: %v", err)
	}

	maxSlots := ckksParams.MaxSlots() // = 2^(logN-1)
	batchSize = int(maxSlots / rowValue)

	babyStep, gaintStep := ChooseSteps(rowValue)
	fmt.Println(babyStep, gaintStep)

	modelParams = &ModelParams{
		NumBatch:   batchSize,      // num_X
		NumRow:     rowValue,       // num_row
		NumCol:     4096,           // num_col
		NumInter:   14336,          // num_inter
		NumLayers:  32,             // num_layers
		NumHeads:   32,             // num_heads
		NumRealRow: rowValue,       // 如果读取矩阵不是128×768，这里设置原始行数的值
		SqrtD:      math.Sqrt(128), // sqrt_d
		ModelPath:  modelPath,      // file path to the model
		BabyStep:   babyStep,       // 示例值
		GiantStep:  gaintStep,      // 示例值

		// 近似Exp的参数
		ExpMinValue: -60,
		ExpMaxValue: 15,
		ExpDegree:   63,

		ConstantValue: 2.08, // 计算softmax时的常数系数

		// 1/x 近似参数
		InvMinValue: 0.03,
		InvMaxValue: 300,
		InvDegree:   255,
		InvIter:     2,

		// RMSNormInput近似的参数
		InvSqrtMinValue1: 0.01,
		InvSqrtMaxValue1: 1,
		InvSqrtDegree1:   31,
		InvSqrtIter1:     2,

		// SiLU近似的参数
		GeluMinValue: -10,
		GeluMaxValue: 10,
		GeluDegree:   127,

		// RMSNormFFNInput近似的参数
		InvSqrtMinValue2: 0.01,
		InvSqrtMaxValue2: 100,
		InvSqrtDegree2:   31,
		InvSqrtIter2:     2,

		NumColKV: 1024,
		GroupQ:   32,
		GroupKV:  8,
	}

	// 检查 ckksParams 的最大slots是否等于 batch*row
	// maxSlots := ckksParams.MaxSlots()
	if maxSlots != modelParams.NumBatch*modelParams.NumRow {
		panic(fmt.Sprintf("ckksParams.MaxSlots()=%d, 但 NumBatch*NumRow=%d，不一致！", maxSlots, modelParams.NumBatch*modelParams.NumRow))
	}

	// 检查 BabyStep*GiantStep 是否大于等于 NumRow
	if modelParams.BabyStep*modelParams.GiantStep < modelParams.NumRow {
		panic(fmt.Sprintf("BabyStep*GiantStep=%d 小于 NumRow=%d，不满足要求！", modelParams.BabyStep*modelParams.GiantStep, modelParams.NumRow))
	}

	return ckksParams, btpParams, modelParams, nil
}

func InitBert(paramType string, rowValue int, modelPath string) (ckksParams ckks.Parameters, btpParams bootstrapping.Parameters, modelParams *ModelParams, err error) {

	var setParamsFunc func() (ckks.Parameters, bootstrapping.Parameters, error)
	var batchSize int
	switch paramType {
	case "tiny":
		setParamsFunc = SetHEParamsTiny
	case "short":
		setParamsFunc = SetHEParamsShort
	case "base":
		setParamsFunc = SetHEParams
	default:
		return ckks.Parameters{}, bootstrapping.Parameters{}, nil, fmt.Errorf("不支持的参数类型: %s", paramType)
	}

	ckksParams, btpParams, err = setParamsFunc()
	if err != nil {
		return ckks.Parameters{}, bootstrapping.Parameters{}, nil, fmt.Errorf("failed to set HE parameters: %v", err)
	}

	maxSlots := ckksParams.MaxSlots() // = 2^(logN-1)
	batchSize = int(maxSlots / rowValue)

	babyStep, gaintStep := ChooseSteps(rowValue)
	fmt.Println(babyStep, gaintStep)

	modelParams = &ModelParams{
		NumBatch:   batchSize, // num_X
		NumRow:     rowValue,  // num_row
		NumCol:     768,       // num_col
		NumInter:   3072,      // num_inter
		NumLayers:  12,        // num_layers
		NumHeads:   12,        // num_heads
		NumRealRow: rowValue,  // 如果读取矩阵不是128×768，这里设置原始行数的值
		SqrtD:      8.0,       // sqrt_d
		ModelPath:  modelPath, // file path to the model
		BabyStep:   babyStep,  // 示例值
		GiantStep:  gaintStep, // 示例值

		// 近似Exp的参数
		ExpMinValue: -60,
		ExpMaxValue: 15,
		ExpDegree:   63,

		ConstantValue: 2.08, // 计算softmax时的常数系数

		// 1/x 近似参数
		// InvMinValue: 0.039,
		// InvMaxValue: 243,
		InvMinValue: 0.02,
		InvMaxValue: 400,
		InvDegree:   255,
		InvIter:     2,

		// layernorm1近似1/sqrt(x)的参数
		InvSqrtMinValue1: 0.001,
		InvSqrtMaxValue1: 2.1,
		InvSqrtDegree1:   31,
		InvSqrtIter1:     2,

		// 近似GELU的参数
		GeluMinValue: -61,
		GeluMaxValue: 136,
		GeluDegree:   255,

		// layernorm2近似1/sqrt(x)的参数
		// InvSqrtMinValue2: 0.10,
		// InvSqrtMaxValue2: 2000.0,
		// InvSqrtDegree2:   31,
		// InvSqrtIter2:     2,

		InvSqrtMinValue2: 0.5,
		InvSqrtMaxValue2: 2500.0,
		InvSqrtDegree2:   31,
		InvSqrtIter2:     2,
	}

	// 检查 ckksParams 的最大slots是否等于 batch*row
	// maxSlots := ckksParams.MaxSlots()
	if maxSlots != modelParams.NumBatch*modelParams.NumRow {
		panic(fmt.Sprintf("ckksParams.MaxSlots()=%d, 但 NumBatch*NumRow=%d，不一致！", maxSlots, modelParams.NumBatch*modelParams.NumRow))
	}

	// 检查 BabyStep*GiantStep 是否大于等于 NumRow
	if modelParams.BabyStep*modelParams.GiantStep < modelParams.NumRow {
		panic(fmt.Sprintf("BabyStep*GiantStep=%d 小于 NumRow=%d，不满足要求！", modelParams.BabyStep*modelParams.GiantStep, modelParams.NumRow))
	}

	return ckksParams, btpParams, modelParams, nil
}

func InitBertTiny(paramType string, rowValue int, modelPath string) (ckksParams ckks.Parameters, btpParams bootstrapping.Parameters, modelParams *ModelParams, err error) {

	var setParamsFunc func() (ckks.Parameters, bootstrapping.Parameters, error)
	var batchSize int
	switch paramType {
	case "tiny":
		setParamsFunc = SetHEParamsTiny
	case "short":
		setParamsFunc = SetHEParamsShort
	case "base":
		setParamsFunc = SetHEParams
	default:
		return ckks.Parameters{}, bootstrapping.Parameters{}, nil, fmt.Errorf("不支持的参数类型: %s", paramType)
	}

	ckksParams, btpParams, err = setParamsFunc()
	if err != nil {
		return ckks.Parameters{}, bootstrapping.Parameters{}, nil, fmt.Errorf("failed to set HE parameters: %v", err)
	}

	maxSlots := ckksParams.MaxSlots() // = 2^(logN-1)
	batchSize = int(maxSlots / rowValue)

	babyStep, gaintStep := ChooseSteps(rowValue)
	// fmt.Println(babyStep, gaintStep)

	// qnli 和 rte
	modelParams = &ModelParams{
		NumBatch:   batchSize, // num_X
		NumRow:     rowValue,  // num_row
		NumCol:     128,       // num_col
		NumInter:   512,       // num_inter
		NumLayers:  2,         // num_layers
		NumHeads:   2,         // num_heads
		NumRealRow: rowValue,  // 如果读取矩阵不是128×768，这里设置原始行数的值
		SqrtD:      8.0,       // sqrt_d
		ModelPath:  modelPath, // file path to the model
		BabyStep:   babyStep,  // 示例值
		GiantStep:  gaintStep, // 示例值

		// 近似Exp的参数
		ExpMinValue: -60,
		ExpMaxValue: 15,
		ExpDegree:   63,

		ConstantValue: 2.08, // bert_tiny_data_qnli/bert_tiny_data_rte/bert_tiny_data_sst2

		// 1/x 近似参数
		InvMinValue: 0.03,
		InvMaxValue: 300,
		InvDegree:   255,
		InvIter:     2,

		// layernorm1近似1/sqrt(x)的参数*d
		InvSqrtMinValue1: 0.01,
		InvSqrtMaxValue1: 6.0,
		InvSqrtDegree1:   31,
		InvSqrtIter1:     2,

		// 近似GELU的参数
		GeluMinValue: -61,
		GeluMaxValue: 136,
		GeluDegree:   255,

		// layernorm2近似1/sqrt(x)的参数
		InvSqrtMinValue2: 0.10,
		InvSqrtMaxValue2: 2000.0,
		InvSqrtDegree2:   31,
		InvSqrtIter2:     2,
	}

	// 检查 ckksParams 的最大slots是否等于 batch*row
	// maxSlots := ckksParams.MaxSlots()
	if maxSlots != modelParams.NumBatch*modelParams.NumRow {
		panic(fmt.Sprintf("ckksParams.MaxSlots()=%d, 但 NumBatch*NumRow=%d，不一致！", maxSlots, modelParams.NumBatch*modelParams.NumRow))
	}

	// 检查 BabyStep*GiantStep 是否大于等于 NumRow
	if modelParams.BabyStep*modelParams.GiantStep < modelParams.NumRow {
		panic(fmt.Sprintf("BabyStep*GiantStep=%d 小于 NumRow=%d，不满足要求！", modelParams.BabyStep*modelParams.GiantStep, modelParams.NumRow))
	}

	return ckksParams, btpParams, modelParams, nil
}

func SetHEParams() (ckksParams ckks.Parameters, btpParams bootstrapping.Parameters, err error) {
	// 1.Set the parameters for the HE scheme
	ckksParams, err = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            16,                                                                // ring degree = 2^16, log QP <= 1748
		LogQ:            []int{58, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45}, // moduli chain
		LogP:            []int{60, 60, 60, 60},                                             // aux moduli for bootstrapping
		LogDefaultScale: 45,
		Xs:              ring.Ternary{H: 192}, // secret key distribution
	})
	if err != nil {
		return ckks.Parameters{}, bootstrapping.Parameters{}, fmt.Errorf("failed to create CKKS parameters: %w", err)
	}

	// 2. Build bootstrapping circuit parameters
	btpLit := bootstrapping.ParametersLiteral{
		LogN: utils.Pointy(ckksParams.LogN()), // match residual LogN
		// LogP: []int{60, 60, 60},               // must match ckksParams.LogP
		LogP: []int{60, 60, 60, 60},
		Xs:   ckksParams.Xs(), // same secret distribution
	}
	btpParams, err = bootstrapping.NewParametersFromLiteral(ckksParams, btpLit)
	if err != nil {
		return ckks.Parameters{}, bootstrapping.Parameters{}, fmt.Errorf("failed to create bootstrapping parameters: %w", err)
	}

	// Print the parameters
	fmt.Printf("CKKS parameters: logN=%d, logSlots=%d, H=%d, sigma=%.2f, logQP=%.2f, levels=%d, scale=2^%d\n",
		btpParams.ResidualParameters.LogN(),
		btpParams.ResidualParameters.LogMaxSlots(),
		btpParams.ResidualParameters.XsHammingWeight(),
		btpParams.ResidualParameters.Xe(),
		ckksParams.LogQP(),
		btpParams.ResidualParameters.MaxLevel(),
		btpParams.ResidualParameters.LogDefaultScale())

	fmt.Println()
	fmt.Printf("Bootstrapping parameters: logN=%d, logSlots=%d, H(%d; %d), sigma=%.2f, logQP=%d, levels=%d, scale=2^%d\n",
		btpParams.BootstrappingParameters.LogN(),
		btpParams.BootstrappingParameters.LogMaxSlots(),
		btpParams.BootstrappingParameters.XsHammingWeight(),
		btpParams.EphemeralSecretWeight,
		btpParams.BootstrappingParameters.Xe(),
		int(btpParams.BootstrappingParameters.LogQP()),
		btpParams.BootstrappingParameters.QCount(),
		btpParams.BootstrappingParameters.LogDefaultScale())

	// Return the parameters
	return ckksParams, btpParams, nil
}

// SetHEParamsShort sets the parameters for a short CKKS scheme with bootstrapping.
// It uses a smaller ring degree and fewer moduli for testing purposes.
func SetHEParamsShort() (ckksParams ckks.Parameters, btpParams bootstrapping.Parameters, err error) {
	// 1.Set the parameters for the HE scheme
	ckksParams, err = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            10,                                                                // ring degree = 2^16, log QP <= 1748
		LogQ:            []int{58, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45}, // moduli chain
		LogP:            []int{60, 60, 60, 60},                                             // aux moduli for bootstrapping
		LogDefaultScale: 45,
		Xs:              ring.Ternary{H: 192}, // secret key distribution
	})
	if err != nil {
		return ckks.Parameters{}, bootstrapping.Parameters{}, fmt.Errorf("failed to create CKKS parameters: %w", err)
	}

	// 2. Build bootstrapping circuit parameters
	btpLit := bootstrapping.ParametersLiteral{
		LogN: utils.Pointy(ckksParams.LogN()), // match residual LogN
		// LogP: []int{60, 60, 60},               // must match ckksParams.LogP
		LogP: []int{60, 60, 60, 60},
		Xs:   ckksParams.Xs(), // same secret distribution
	}
	btpParams, err = bootstrapping.NewParametersFromLiteral(ckksParams, btpLit)
	if err != nil {
		return ckks.Parameters{}, bootstrapping.Parameters{}, fmt.Errorf("failed to create bootstrapping parameters: %w", err)
	}

	// Print the parameters
	fmt.Printf("CKKS parameters: logN=%d, logSlots=%d, H=%d, sigma=%.2f, logQP=%.2f, levels=%d, scale=2^%d\n",
		btpParams.ResidualParameters.LogN(),
		btpParams.ResidualParameters.LogMaxSlots(),
		btpParams.ResidualParameters.XsHammingWeight(),
		btpParams.ResidualParameters.Xe(),
		ckksParams.LogQP(),
		btpParams.ResidualParameters.MaxLevel(),
		btpParams.ResidualParameters.LogDefaultScale())

	fmt.Println()
	fmt.Printf("Bootstrapping parameters: logN=%d, logSlots=%d, H(%d; %d), sigma=%.2f, logQP=%d, levels=%d, scale=2^%d\n",
		btpParams.BootstrappingParameters.LogN(),
		btpParams.BootstrappingParameters.LogMaxSlots(),
		btpParams.BootstrappingParameters.XsHammingWeight(),
		btpParams.EphemeralSecretWeight,
		btpParams.BootstrappingParameters.Xe(),
		int(btpParams.BootstrappingParameters.LogQP()),
		btpParams.BootstrappingParameters.QCount(),
		btpParams.BootstrappingParameters.LogDefaultScale())

	// Return the parameters
	return ckksParams, btpParams, nil
}

// SetHEParamsShort sets the parameters for a short CKKS scheme with bootstrapping.
// It uses a smaller ring degree and fewer moduli for testing purposes.
func SetHEParamsTiny() (ckksParams ckks.Parameters, btpParams bootstrapping.Parameters, err error) {
	// 1.Set the parameters for the HE scheme
	ckksParams, err = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            6,                                                                 // ring degree = 2^16, log QP <= 1748
		LogQ:            []int{58, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45}, // moduli chain
		LogP:            []int{60, 60, 60, 60},                                             // aux moduli for bootstrapping
		LogDefaultScale: 45,
		Xs:              ring.Ternary{H: 192}, // secret key distribution
	})
	if err != nil {
		return ckks.Parameters{}, bootstrapping.Parameters{}, fmt.Errorf("failed to create CKKS parameters: %w", err)
	}

	// 2. Build bootstrapping circuit parameters
	btpLit := bootstrapping.ParametersLiteral{
		LogN: utils.Pointy(ckksParams.LogN()), // match residual LogN
		// LogP: []int{60, 60, 60},               // must match ckksParams.LogP
		LogP: []int{60, 60, 60, 60},
		Xs:   ckksParams.Xs(), // same secret distribution
	}
	btpParams, err = bootstrapping.NewParametersFromLiteral(ckksParams, btpLit)
	if err != nil {
		return ckks.Parameters{}, bootstrapping.Parameters{}, fmt.Errorf("failed to create bootstrapping parameters: %w", err)
	}

	// Print the parameters
	fmt.Printf("CKKS parameters: logN=%d, logSlots=%d, H=%d, sigma=%.2f, logQP=%.2f, levels=%d, scale=2^%d\n",
		btpParams.ResidualParameters.LogN(),
		btpParams.ResidualParameters.LogMaxSlots(),
		btpParams.ResidualParameters.XsHammingWeight(),
		btpParams.ResidualParameters.Xe(),
		ckksParams.LogQP(),
		btpParams.ResidualParameters.MaxLevel(),
		btpParams.ResidualParameters.LogDefaultScale())

	fmt.Println()
	fmt.Printf("Bootstrapping parameters: logN=%d, logSlots=%d, H(%d; %d), sigma=%.2f, logQP=%d, levels=%d, scale=2^%d\n",
		btpParams.BootstrappingParameters.LogN(),
		btpParams.BootstrappingParameters.LogMaxSlots(),
		btpParams.BootstrappingParameters.XsHammingWeight(),
		btpParams.EphemeralSecretWeight,
		btpParams.BootstrappingParameters.Xe(),
		int(btpParams.BootstrappingParameters.LogQP()),
		btpParams.BootstrappingParameters.QCount(),
		btpParams.BootstrappingParameters.LogDefaultScale())

	// Return the parameters
	return ckksParams, btpParams, nil
}

func ChooseSteps(numRow int) (baby, giant int) {
	if numRow <= 0 {
		return 1, 1
	}
	// 1) 初始 baby = ceil(sqrt(numRow))
	baby = int(math.Ceil(math.Sqrt(float64(numRow))))

	// 2) giant = ceil(numRow / baby)
	giant = int(math.Ceil(float64(numRow) / float64(baby)))

	// 3) 如果 giant 仍比 baby 大，向上调 baby
	if giant > baby {
		baby++
		giant = int(math.Ceil(float64(numRow) / float64(baby)))
	}
	// 4) 此时必有 giant <= baby 且差值<=1
	return baby, giant
}
