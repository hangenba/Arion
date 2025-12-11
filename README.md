# Arion: Privacy-Preserving Transformer Inference Framework

**Arion** is a high-performance framework for privacy-preserving machine learning inference using Fully Homomorphic Encryption (FHE). It is specifically designed to accelerate the inference of Transformer-based models (e.g., BERT-Base) on encrypted data.

Core to Arion is the **Double Baby-Step Giant-Step (Double-BSGS)** algorithm, which significantly reduces the computational complexity of attention mechanisms from linear to square root scaling ($O(\sqrt{m})$).

## 🚀 Key Features

* **Optimized Attention Mechanism**: Implements the Double-BSGS algorithm to minimize rotation keys and computational overhead in the Attention layer.
* **End-to-End BERT Inference**: Supports fully encrypted inference for BERT-Tiny and BERT-Base architectures.
* **Multi-Threading Support**: Leveraging Go's concurrency with **64 threads** to execute high-performance parallel matrix operations .
* **Flexible Security Parameters**: Pre-configured settings for various testing scales:
    * **Tiny**: `logN=6` (Fast debugging)
    * **Short**: `logN=10` (Correctness verification)
    * **Base**: `logN=16` (Production-grade security & accuracy)

## 🛠️ Prerequisites

* **Go**: Version 1.24.5 or higher
* **Hardware Requirements**:
    * For **Tiny/Short** parameters: Standard CPU/RAM, At least**20GB+ RAM**.
    * For **Base** parameters ($N=2^{16}$): At least **600GB+ RAM** is recommended due to large key sizes.

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/hangenba/Arion.git
    cd Arion
    ```

2.  **Install dependencies**
    ```bash
    go mod tidy
    ```

3.  **Prepare Model Data**
    Ensure your pre-trained model weights and input samples are placed in the directory (e.g., `bert_base_data`, `bert_tiny_data`).
    **Tip:** After end-to-end execution, all inference results and benchmark logs are automatically saved to the `output/` directory.

## 🏃 Usage

Arion provides an interactive Command Line Interface (CLI) to simplify benchmarking and testing.

Run the main entry point:

```bash
go run main.go
```