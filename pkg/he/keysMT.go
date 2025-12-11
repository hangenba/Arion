package he

import (
	"Arion/configs"
	"fmt"
	"runtime"
	"sync"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func CopyKeyGenerator(params rlwe.ParameterProvider, enc *rlwe.Encryptor) *rlwe.KeyGenerator {
	return &rlwe.KeyGenerator{
		Encryptor: enc,
	}
}

func GenerateKeysAndBtsKeysMT(ckksParams ckks.Parameters, btpParams bootstrapping.Parameters, modelParams *configs.ModelParams, numThreads int) (*ckks.Encoder, *rlwe.Encryptor, *ckks.Evaluator, *rlwe.Decryptor, *bootstrapping.Evaluator) {
	fmt.Println("Key generation with parallel GaloisKeys...")

	kgen := ckks.NewKeyGenerator(ckksParams)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk) // Note that we can generate any number of public keys associated to the same Secret Key.
	rlk := kgen.GenRelinearizationKeyNew(sk)
	ecd := ckks.NewEncoder(ckksParams)
	enc := ckks.NewEncryptor(ckksParams, pk)

	dec := rlwe.NewDecryptor(ckksParams, sk)

	// evk := rlwe.NewMemEvaluationKeySet(rlk) // Removed unused variable

	// Step 1: GaloisKeys并行生成
	LogSlots := ckksParams.LogMaxSlots()
	Slots := 1 << LogSlots
	rangSlots := Slots / modelParams.NumBatch

	var rotNumbers []int
	for i := 1; i <= rangSlots; i++ {
		rotNumbers = append(rotNumbers, i*modelParams.NumBatch)
		rotNumbers = append(rotNumbers, -i*modelParams.NumBatch)
	}

	galEls := ckksParams.GaloisElements(rotNumbers)

	galoisKeys := ParallelGenGaloisKeys(ckksParams, sk, galEls, numThreads)
	evk := rlwe.NewMemEvaluationKeySet(rlk, galoisKeys...)
	eval := ckks.NewEvaluator(ckksParams, evk)

	fmt.Println("GaloisKeys generation completed.")

	// Step 2: Bootstrapping Key
	btpEvk, _, err := GenEvaluationKeysParallel(btpParams, sk, numThreads)
	if err != nil {
		panic(err)
	}
	fmt.Println("Bootstrapping GenEvaluationKeys Completed.")
	btpEval, evalErr := bootstrapping.NewEvaluator(btpParams, btpEvk)
	if evalErr != nil {
		panic(evalErr)
	}

	fmt.Println("Bootstrapping Key Generation Completed.")

	return ecd, enc, eval, dec, btpEval
}

func GenerateKeysAndBtsKeysAndMultiEncMT(ckksParams ckks.Parameters, btpParams bootstrapping.Parameters, modelParams *configs.ModelParams, numThreads int) (*ckks.Encoder, *rlwe.Encryptor, []*ckks.Evaluator, *rlwe.Decryptor, *bootstrapping.Evaluator) {
	fmt.Println("Key generation with parallel GaloisKeys...")

	kgen := ckks.NewKeyGenerator(ckksParams)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk) // Note that we can generate any number of public keys associated to the same Secret Key.
	rlk := kgen.GenRelinearizationKeyNew(sk)
	ecd := ckks.NewEncoder(ckksParams)
	enc := ckks.NewEncryptor(ckksParams, pk)

	dec := rlwe.NewDecryptor(ckksParams, sk)

	// evk := rlwe.NewMemEvaluationKeySet(rlk) // Removed unused variable

	// Step 1: GaloisKeys并行生成
	LogSlots := ckksParams.LogMaxSlots()
	Slots := 1 << LogSlots
	rangSlots := Slots / modelParams.NumBatch

	var rotNumbers []int
	for i := 1; i <= rangSlots; i++ {
		rotNumbers = append(rotNumbers, i*modelParams.NumBatch)
		rotNumbers = append(rotNumbers, -i*modelParams.NumBatch)
	}

	galEls := ckksParams.GaloisElements(rotNumbers)
	evalM := make([]*ckks.Evaluator, modelParams.NumHeads)
	for i := 0; i < modelParams.NumHeads; i++ {
		galoisKeys := ParallelGenGaloisKeys(ckksParams, sk, galEls, numThreads)
		evk := rlwe.NewMemEvaluationKeySet(rlk, galoisKeys...)
		eval := ckks.NewEvaluator(ckksParams, evk)
		evalM[i] = eval
	}

	fmt.Println("GaloisKeys generation completed.")

	// Step 2: Bootstrapping Key
	btpEvk, _, err := GenEvaluationKeysParallel(btpParams, sk, numThreads)
	if err != nil {
		panic(err)
	}
	fmt.Println("Bootstrapping GenEvaluationKeys Completed.")
	btpEval, evalErr := bootstrapping.NewEvaluator(btpParams, btpEvk)
	if evalErr != nil {
		panic(evalErr)
	}

	fmt.Println("Bootstrapping Key Generation Completed.")

	return ecd, enc, evalM, dec, btpEval
}

func ParallelGenGaloisKeys(params rlwe.ParameterProvider, sk *rlwe.SecretKey, galEls []uint64, numThreads int) []*rlwe.GaloisKey {
	galoisKeys := make([]*rlwe.GaloisKey, len(galEls))
	chunkSize := (len(galEls) + numThreads - 1) / numThreads // ceil(len / numThreads)

	var wg sync.WaitGroup
	runtime.GOMAXPROCS(numThreads) // 设置线程数（可选，不写的话默认GOMAXPROCS）

	for t := 0; t < numThreads; t++ {
		start := t * chunkSize
		end := (t + 1) * chunkSize
		if end > len(galEls) {
			end = len(galEls)
		}

		wg.Add(1)
		go func(start, end, threadNum int) {
			defer wg.Done()

			// 每个线程创建一个新的 KeyGenerator（无状态，线程安全）
			localKgen := rlwe.NewKeyGenerator(params)

			for i := start; i < end; i++ {
				galEl := galEls[i]

				// 生成新的 GaloisKey 并保存
				galoisKeys[i] = localKgen.GenGaloisKeyNew(galEl, sk)
			}
		}(start, end, t)
	}

	wg.Wait()
	return galoisKeys
}

func GenEvaluationKeysParallel(p bootstrapping.Parameters, skN1 *rlwe.SecretKey, numThreads int) (btpkeys *bootstrapping.EvaluationKeys, skN2 *rlwe.SecretKey, err error) {

	var EvkN1ToN2, EvkN2ToN1 *rlwe.EvaluationKey
	var EvkRealToCmplx *rlwe.EvaluationKey
	var EvkCmplxToReal *rlwe.EvaluationKey
	paramsN2 := p.BootstrappingParameters

	kgen := rlwe.NewKeyGenerator(paramsN2)

	if p.ResidualParameters.N() != paramsN2.N() {
		skN2 = kgen.GenSecretKeyNew()

		if p.ResidualParameters.RingType() == ring.ConjugateInvariant {
			EvkCmplxToReal, EvkRealToCmplx = kgen.GenEvaluationKeysForRingSwapNew(skN2, skN1)
		} else {
			EvkN1ToN2 = kgen.GenEvaluationKeyNew(skN1, skN2)
			EvkN2ToN1 = kgen.GenEvaluationKeyNew(skN2, skN1)
		}

	} else {

		ringQ := paramsN2.RingQ()
		ringP := paramsN2.RingP()

		skN2 = rlwe.NewSecretKey(paramsN2)
		buff := ringQ.NewPoly()

		rlwe.ExtendBasisSmallNormAndCenterNTTMontgomery(ringQ, ringQ, skN1.Value.Q, buff, skN2.Value.Q)
		rlwe.ExtendBasisSmallNormAndCenterNTTMontgomery(ringQ, ringP, skN1.Value.Q, buff, skN2.Value.P)
	}

	EvkDenseToSparse, EvkSparseToDense := GenEncapsulationEvaluationKeys(p, skN2)

	// 并行生成 GaloisKeys
	galEls := append(p.GaloisElements(paramsN2), paramsN2.GaloisElementForComplexConjugation())
	galoisKeys := ParallelGenGaloisKeys(paramsN2, skN2, galEls, numThreads)

	// Relinearization Key
	rlk := kgen.GenRelinearizationKeyNew(skN2)

	// 汇总 EvaluationKeys
	return &bootstrapping.EvaluationKeys{
		EvkN1ToN2:           EvkN1ToN2,
		EvkN2ToN1:           EvkN2ToN1,
		EvkRealToCmplx:      EvkRealToCmplx,
		EvkCmplxToReal:      EvkCmplxToReal,
		MemEvaluationKeySet: rlwe.NewMemEvaluationKeySet(rlk, galoisKeys...),
		EvkDenseToSparse:    EvkDenseToSparse,
		EvkSparseToDense:    EvkSparseToDense,
	}, skN2, nil
}

func GenEncapsulationEvaluationKeys(p bootstrapping.Parameters, skDense *rlwe.SecretKey) (EvkDenseToSparse, EvkSparseToDense *rlwe.EvaluationKey) {
	params := p.BootstrappingParameters

	if p.EphemeralSecretWeight == 0 {
		return
	}

	paramsSparse, _ := rlwe.NewParametersFromLiteral(rlwe.ParametersLiteral{
		LogN: params.LogN(),
		Q:    params.Q()[:1],
		P:    params.P()[:1],
	})

	kgenSparse := rlwe.NewKeyGenerator(paramsSparse)
	kgenDense := rlwe.NewKeyGenerator(params)
	skSparse := kgenSparse.GenSecretKeyWithHammingWeightNew(p.EphemeralSecretWeight)

	EvkDenseToSparse = kgenSparse.GenEvaluationKeyNew(skDense, skSparse)
	EvkSparseToDense = kgenDense.GenEvaluationKeyNew(skSparse, skDense)
	return
}
