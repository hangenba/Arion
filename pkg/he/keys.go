package he

import (
	"Arion/configs"
	"fmt"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func GenerateKeys(ckksParams ckks.Parameters, modelParams *configs.ModelParams) (*ckks.Encoder, *rlwe.Encryptor, *ckks.Evaluator, *rlwe.Decryptor) {
	// This function is a placeholder for the key generation logic.
	// It should be implemented with the actual logic for generating keys.
	fmt.Println("CKKS initialization and key generation...")

	kgen := ckks.NewKeyGenerator(ckksParams)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk) // Note that we can generate any number of public keys associated to the same Secret Key.
	rlk := kgen.GenRelinearizationKeyNew(sk)
	ecd := ckks.NewEncoder(ckksParams)
	enc := ckks.NewEncryptor(ckksParams, pk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	eval := ckks.NewEvaluator(ckksParams, evk)
	dec := rlwe.NewDecryptor(ckksParams, sk)

	LogSlots := ckksParams.LogMaxSlots()
	Slots := 1 << LogSlots
	rangSlots := Slots / modelParams.NumBatch

	// 生成旋转步数
	var rotNumbers []int
	for i := 1; i <= rangSlots; i++ {
		rotNumbers = append(rotNumbers, i*modelParams.NumBatch)
		rotNumbers = append(rotNumbers, -i*modelParams.NumBatch)
	}
	galEls := ckksParams.GaloisElements(rotNumbers)
	eval = eval.WithKey(rlwe.NewMemEvaluationKeySet(rlk, kgen.GenGaloisKeysNew(galEls, sk)...))
	fmt.Println("CKKS initialization and key generation completed.")

	return ecd, enc, eval, dec
}

func GenerateKeysAndBtsKeys(ckksParams ckks.Parameters, btpParams bootstrapping.Parameters, modelParams *configs.ModelParams) (*ckks.Encoder, *rlwe.Encryptor, *ckks.Evaluator, *rlwe.Decryptor, *bootstrapping.Evaluator) {
	// This function is a placeholder for the key generation logic.
	// It should be implemented with the actual logic for generating keys.
	fmt.Println("CKKS initialization and key generation...")

	kgen := ckks.NewKeyGenerator(ckksParams)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk) // Note that we can generate any number of public keys associated to the same Secret Key.
	rlk := kgen.GenRelinearizationKeyNew(sk)
	ecd := ckks.NewEncoder(ckksParams)
	enc := ckks.NewEncryptor(ckksParams, pk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	eval := ckks.NewEvaluator(ckksParams, evk)
	dec := rlwe.NewDecryptor(ckksParams, sk)

	LogSlots := ckksParams.LogMaxSlots()
	Slots := 1 << LogSlots
	rangSlots := Slots / modelParams.NumBatch

	// 生成旋转步数
	var rotNumbers []int
	for i := 1; i <= rangSlots; i++ {
		rotNumbers = append(rotNumbers, i*modelParams.NumBatch)
		rotNumbers = append(rotNumbers, -i*modelParams.NumBatch)
	}
	galEls := ckksParams.GaloisElements(rotNumbers)
	eval = eval.WithKey(rlwe.NewMemEvaluationKeySet(rlk, kgen.GenGaloisKeysNew(galEls, sk)...))
	fmt.Println("CKKS initialization and key generation completed.")

	fmt.Println()
	fmt.Println("Generating bootstrapping evaluation keys...")
	btpEvk, _, err := btpParams.GenEvaluationKeys(sk)
	if err != nil {
		panic(err)
	}
	fmt.Println("Done")
	// Instantiates the bootstrapper
	var btpEval *bootstrapping.Evaluator
	if btpEval, err = bootstrapping.NewEvaluator(btpParams, btpEvk); err != nil {
		panic(err)
	}

	return ecd, enc, eval, dec, btpEval
}
