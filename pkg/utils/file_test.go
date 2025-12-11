package utils

import (
	"os"
	"testing"
)

func TestReadCSVToMatrix(t *testing.T) {
	// 创建临时CSV文件
	tmpFile, err := os.CreateTemp("", "test_*.csv")
	if err != nil {
		t.Fatalf("无法创建临时文件: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	csvContent := "1.0,2.0,3.0\n4.0,5.0,6.0"
	if _, err := tmpFile.WriteString(csvContent); err != nil {
		t.Fatalf("写入临时文件失败: %v", err)
	}
	tmpFile.Close()

	mat, err := ReadCSVToMatrix(tmpFile.Name())
	if err != nil {
		t.Fatalf("ReadCSVToMatrix 返回错误: %v", err)
	}
	rows, cols := mat.Dims()
	if rows != 2 || cols != 3 {
		t.Errorf("期望矩阵维度为2x3，实际为%dx%d", rows, cols)
	}
	want := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if mat.At(i, j) != want[i*cols+j] {
				t.Errorf("mat[%d,%d]=%v, 期望%v", i, j, mat.At(i, j), want[i*cols+j])
			}
		}
	}
}
