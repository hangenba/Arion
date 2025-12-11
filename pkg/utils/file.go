package utils

import (
	"Arion/configs"
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

/*
file.go 提供了模型参数文件的读取与处理工具。

本文件主要功能包括：
- 定义 LayerParameters 结构体，用于 BERT 模型每一层的所有权重和偏置参数（以矩阵形式）。
- 提供 ReadCSVToMatrix 函数，将 CSV 文件读取为 gonum 的 Dense 矩阵。
- 提供 ReadInput 函数，根据 ModelParams 拼接路径并读取输入矩阵。
- 提供 ReadLayerParameters 函数，按结构体字段顺序批量读取指定层的所有参数文件，并返回 LayerParameters 实例。
  **特别说明：在读取 LayerAttentionSelfQueryWeight 和 LayerAttentionSelfQueryBias 时，自动对其数值除以 SqrtD（即根号d），以满足 BERT 注意力机制的缩放要求。**
- 提供 PrintLayerParametersDims 函数，美观打印 LayerParameters 中所有参数矩阵的维度信息，便于调试和模型结构检查。

所有路径拼接均使用 filepath.Join，保证跨平台兼容性。
*/

type LayerParameters struct {
	// Add fields as per your requirements
	LayerNumber int // Layer number, e.g., 0 for the first layer

	LayerAttentionSelfQueryWeight mat.Matrix // 768 X 768
	LayerAttentionSelfQueryBias   mat.Matrix // 768
	LayerAttentionSelfKeyWeight   mat.Matrix //768 X 768
	LayerAttentionSelfKeyBias     mat.Matrix // 768
	LayerAttentionSelfValueWeight mat.Matrix // 768 X 768
	LayerAttentionSelfValueBias   mat.Matrix // 768

	LayerAttentionOutputDenseWeight     mat.Matrix // 768 X 768
	LayerAttentionOutputDenseBias       mat.Matrix // 768
	LayerAttentionOutputLayerNormWeight mat.Matrix // 768
	LayerAttentionOutputLayerNormBias   mat.Matrix // 768

	LayerIntermediateDenseWeight mat.Matrix // 3072 X 768
	LayerIntermediateDenseBias   mat.Matrix // 3072
	LayerOutputDenseWeight       mat.Matrix // 768 X 3072
	LayerOutputDenseBias         mat.Matrix // 768
	LayerOutputLayerNormWeight   mat.Matrix // 768
	LayerOutputLayerNormBias     mat.Matrix // 768
}

// ReadCSVToMatrix 读取CSV文件并返回gonum的Dense矩阵
func ReadCSVToMatrix(filename string) (*mat.Dense, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("无法打开文件: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1 // 允许每行不同列数，不严格限制

	var data [][]float64

	for {
		record, err := reader.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			return nil, fmt.Errorf("读取 CSV 行失败: %w", err)
		}

		row := make([]float64, len(record))
		for i, field := range record {
			val, err := strconv.ParseFloat(strings.TrimSpace(field), 64)
			if err != nil {
				return nil, fmt.Errorf("解析数字失败: %w", err)
			}
			row[i] = val
		}
		data = append(data, row)
	}

	// shape 检查
	if len(data) == 0 {
		return nil, fmt.Errorf("CSV文件为空")
	}
	rows := len(data)
	cols := len(data[0])

	// flatten
	flat := make([]float64, 0, rows*cols)
	for _, row := range data {
		if len(row) != cols {
			return nil, fmt.Errorf("CSV 每行列数不一致")
		}
		flat = append(flat, row...)
	}

	return mat.NewDense(rows, cols, flat), nil
}

// ReadInput 根据模型参数中的 ModelPath 拼接文件名并读取 CSV 文件为矩阵
func ReadInput(params *configs.ModelParams, filename string) (*mat.Dense, error) {
	inputPath := filepath.Join(params.ModelPath, "layer_0", "Attention", "BertSelfAttention", "allresults", filename)
	return ReadCSVToMatrix(inputPath)
}

// ReadLayerParameters 读取指定层的所有参数，返回 LayerParameters
func ReadLayerParameters(params *configs.ModelParams, layer int) (*LayerParameters, error) {
	var err error

	bertSelfAttentionPath := filepath.Join(params.ModelPath, fmt.Sprintf("layer_%d", layer), "Attention", "BertSelfAttention", "parms")
	selfAttentionOutputPath := filepath.Join(params.ModelPath, fmt.Sprintf("layer_%d", layer), "Attention", "SelfOutput", "parms")
	intermediatePath := filepath.Join(params.ModelPath, fmt.Sprintf("layer_%d", layer), "Intermediate", "parms")
	outputPath := filepath.Join(params.ModelPath, fmt.Sprintf("layer_%d", layer), "Output", "parms")

	lp := &LayerParameters{
		LayerNumber: layer,
	}

	// 顺序严格按照结构体字段顺序
	layerAttentionSelfQueryWeight, err := ReadCSVToMatrix(filepath.Join(bertSelfAttentionPath, "query_weight.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w", filepath.Join(bertSelfAttentionPath, "query_weight.csv"), err)
	}
	layerAttentionSelfQueryBias, err := ReadCSVToMatrix(filepath.Join(bertSelfAttentionPath, "query_bias.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w", filepath.Join(bertSelfAttentionPath, "query_bias.csv"), err)
	}
	// 注意：BERT模型中，query的权重和偏置矩阵都需要除以 sqrt_d
	lp.LayerAttentionSelfQueryWeight = ScaleMatrix(layerAttentionSelfQueryWeight, 1/params.SqrtD)
	lp.LayerAttentionSelfQueryBias = ScaleMatrix(layerAttentionSelfQueryBias, 1/params.SqrtD)

	lp.LayerAttentionSelfKeyWeight, err = ReadCSVToMatrix(filepath.Join(bertSelfAttentionPath, "key_weight.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w", filepath.Join(bertSelfAttentionPath, "key_weight.csv"), err)
	}
	lp.LayerAttentionSelfKeyBias, err = ReadCSVToMatrix(filepath.Join(bertSelfAttentionPath, "key_bias.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w", filepath.Join(bertSelfAttentionPath, "key_bias.csv"), err)
	}
	lp.LayerAttentionSelfValueWeight, err = ReadCSVToMatrix(filepath.Join(bertSelfAttentionPath, "value_weight.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w", filepath.Join(bertSelfAttentionPath, "value_weight.csv"), err)
	}
	lp.LayerAttentionSelfValueBias, err = ReadCSVToMatrix(filepath.Join(bertSelfAttentionPath, "value_bias.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w", filepath.Join(bertSelfAttentionPath, "value_bias.csv"), err)
	}

	lp.LayerAttentionOutputDenseWeight, err = ReadCSVToMatrix(filepath.Join(selfAttentionOutputPath, "self_output_dense_weight.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w", filepath.Join(selfAttentionOutputPath, "self_output_dense_weight.csv"), err)
	}
	lp.LayerAttentionOutputDenseBias, err = ReadCSVToMatrix(filepath.Join(selfAttentionOutputPath, "self_output_dense_bias.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w", filepath.Join(selfAttentionOutputPath, "self_output_dense_bias.csv"), err)
	}
	lp.LayerAttentionOutputLayerNormWeight, err = ReadCSVToMatrix(filepath.Join(selfAttentionOutputPath, "self_output_LayerNorm_weight.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w", filepath.Join(selfAttentionOutputPath, "self_output_LayerNorm_weight.csv"), err)
	}
	lp.LayerAttentionOutputLayerNormBias, err = ReadCSVToMatrix(filepath.Join(selfAttentionOutputPath, "self_output_LayerNorm_bias.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w", filepath.Join(selfAttentionOutputPath, "self_output_LayerNorm_bias.csv"), err)
	}

	lp.LayerIntermediateDenseWeight, err = ReadCSVToMatrix(filepath.Join(intermediatePath, "intermediate_dense_weight.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w", filepath.Join(intermediatePath, "intermediate_dense_weight.csv"), err)
	}
	lp.LayerIntermediateDenseBias, err = ReadCSVToMatrix(filepath.Join(intermediatePath, "intermediate_dense_bias.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w", filepath.Join(intermediatePath, "intermediate_dense_bias.csv"), err)
	}

	lp.LayerOutputDenseWeight, err = ReadCSVToMatrix(filepath.Join(outputPath, "final_output_dense_weight.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w", filepath.Join(outputPath, "final_output_dense_weight.csv"), err)
	}
	lp.LayerOutputDenseBias, err = ReadCSVToMatrix(filepath.Join(outputPath, "final_output_dense_bias.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w", filepath.Join(outputPath, "final_output_dense_bias.csv"), err)
	}
	lp.LayerOutputLayerNormWeight, err = ReadCSVToMatrix(filepath.Join(outputPath, "final_output_LayerNorm_weight.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w", filepath.Join(outputPath, "final_output_LayerNorm_weight.csv"), err)
	}
	lp.LayerOutputLayerNormBias, err = ReadCSVToMatrix(filepath.Join(outputPath, "final_output_LayerNorm_bias.csv"))
	if err != nil {
		return nil, fmt.Errorf("读取文件 %s 失败: %w", filepath.Join(outputPath, "final_output_LayerNorm_bias.csv"), err)
	}

	return lp, nil
}

// PrintLayerParametersDims 按结构体顺序美观打印所有矩阵的维度信息
func PrintLayerParametersDims(lp *LayerParameters) {
	printMatrixDims := func(name string, m mat.Matrix) {
		if m == nil {
			fmt.Printf("  %-36s : nil\n", name)
			return
		}
		r, c := m.Dims()
		fmt.Printf("  %-36s : %4d x %-4d\n", name, r, c)
	}

	fmt.Printf("======================= Layer %d 参数维度 =======================\n", lp.LayerNumber)
	fmt.Println("[Attention - Self]")
	printMatrixDims("LayerAttentionSelfQueryWeight", lp.LayerAttentionSelfQueryWeight)
	printMatrixDims("LayerAttentionSelfQueryBias", lp.LayerAttentionSelfQueryBias)
	printMatrixDims("LayerAttentionSelfKeyWeight", lp.LayerAttentionSelfKeyWeight)
	printMatrixDims("LayerAttentionSelfKeyBias", lp.LayerAttentionSelfKeyBias)
	printMatrixDims("LayerAttentionSelfValueWeight", lp.LayerAttentionSelfValueWeight)
	printMatrixDims("LayerAttentionSelfValueBias", lp.LayerAttentionSelfValueBias)

	fmt.Println("[Attention - Output]")
	printMatrixDims("LayerAttentionOutputDenseWeight", lp.LayerAttentionOutputDenseWeight)
	printMatrixDims("LayerAttentionOutputDenseBias", lp.LayerAttentionOutputDenseBias)
	printMatrixDims("LayerAttentionOutputLayerNormWeight", lp.LayerAttentionOutputLayerNormWeight)
	printMatrixDims("LayerAttentionOutputLayerNormBias", lp.LayerAttentionOutputLayerNormBias)

	fmt.Println("[Intermediate]")
	printMatrixDims("LayerIntermediateDenseWeight", lp.LayerIntermediateDenseWeight)
	printMatrixDims("LayerIntermediateDenseBias", lp.LayerIntermediateDenseBias)

	fmt.Println("[Output]")
	printMatrixDims("LayerOutputDenseWeight", lp.LayerOutputDenseWeight)
	printMatrixDims("LayerOutputDenseBias", lp.LayerOutputDenseBias)
	printMatrixDims("LayerOutputLayerNormWeight", lp.LayerOutputLayerNormWeight)
	printMatrixDims("LayerOutputLayerNormBias", lp.LayerOutputLayerNormBias)
	fmt.Println("============================ End ================================")
}

// ReadLayerParameters 读取指定层的所有参数，返回 LayerParameters
func ReadPoolerParameters(params *configs.ModelParams) (*mat.Dense, *mat.Dense, error) {
	var err error

	poolerPath := filepath.Join(params.ModelPath, "pooler", "parms")

	// 顺序严格按照结构体字段顺序
	poolerDenseWeight, err := ReadCSVToMatrix(filepath.Join(poolerPath, "pooler_dense_weight.csv"))
	if err != nil {
		return nil, nil, fmt.Errorf("读取文件 %s 失败: %w", filepath.Join(poolerPath, "pooler_dense_weight.csv"), err)
	}
	poolerDenseBias, err := ReadCSVToMatrix(filepath.Join(poolerPath, "pooler_dense_bias.csv"))
	if err != nil {
		return nil, nil, fmt.Errorf("读取文件 %s 失败: %w", filepath.Join(poolerPath, "pooler_dense_bias.csv"), err)
	}

	return poolerDenseWeight, poolerDenseBias, nil
}

// ReadLayerParameters 读取指定层的所有参数，返回 LayerParameters
func ReadClassifierParameters(params *configs.ModelParams) (*mat.Dense, *mat.Dense, error) {
	var err error

	poolerPath := filepath.Join(params.ModelPath, "classifier", "parms")

	// 顺序严格按照结构体字段顺序
	classifierWeight, err := ReadCSVToMatrix(filepath.Join(poolerPath, "classifier_weight.csv"))
	if err != nil {
		return nil, nil, fmt.Errorf("读取文件 %s 失败: %w", filepath.Join(poolerPath, "classifier_weight.csv"), err)
	}
	classifierBias, err := ReadCSVToMatrix(filepath.Join(poolerPath, "classifier_bias.csv"))
	if err != nil {
		return nil, nil, fmt.Errorf("读取文件 %s 失败: %w", filepath.Join(poolerPath, "classifier_bias.csv"), err)
	}

	return classifierWeight, classifierBias, nil
}

// SaveMatrixToCSV 将矩阵保存为CSV文件
// SaveMatrixToCSV 将矩阵保存为CSV文件
func SaveMatrixToCSV(m mat.Matrix, path string) error {
	r, c := m.Dims()
	// 确保父目录存在
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		return err
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	defer writer.Flush()

	for i := 0; i < r; i++ {
		row := make([]string, c)
		for j := 0; j < c; j++ {
			row[j] = strconv.FormatFloat(m.At(i, j), 'g', -1, 64)
		}
		if err := writer.Write(row); err != nil {
			return err
		}
	}
	return nil
}
