# Low-Resource-Fine-Tuning-and-Deployment-of-Large-Language-Models-using-QLoRA
This project focuses on the efficient fine-tuning of Large Language Models (LLMs) using the QLoRA (Quantized Low-Rank Adaptation) technique. Traditional fine-tuning of LLMs requires high computational resources and large GPU memory, which limits accessibility. To address this challenge, QLoRA enables parameter-efficient training by combining low-rank adaptation with 4-bit quantization, significantly reducing memory usage while maintaining model performance.

In this work, a pre-trained language model was fine-tuned on a custom dataset using QLoRA within a Kaggle Notebook environment powered by an NVIDIA Tesla P100 GPU. The model was optimized for both performance and resource efficiency, making it suitable for deployment in low-resource environments. Additionally, the project explored model optimization techniques and efficient inference strategies.

This approach demonstrates that high-quality LLM fine-tuning can be achieved even on limited hardware, enabling scalable and cost-effective AI development.

🎯 Objective
To implement parameter-efficient fine-tuning of LLMs using QLoRA
To reduce memory and computational requirements during training
To fine-tune a pre-trained model on a custom dataset
To perform training using limited hardware (Kaggle P100 GPU)
To optimize the model for efficient inference and deployment
⚙️ Tools & Technologies
Python
PyTorch
Hugging Face Transformers
PEFT (QLoRA)
BitsAndBytes (4-bit quantization)
Kaggle Notebook
NVIDIA Tesla P100 GPU
🚀 Outcome / Results
Successfully fine-tuned a large language model using QLoRA
Achieved significant memory reduction compared to full fine-tuning
Enabled training on limited GPU resources (P100)
Produced an optimized model suitable for efficient inference<img width="222" height="635" alt="model_architecture" src="https://github.com/user-attachments/assets/a5fdf7e0-8424-4bc3-9e09-14ca6dc15daf" />
🧠 Model Architecture

This project uses a Qwen-based Large Language Model (LLM) fine-tuned using QLoRA (Quantized Low-Rank Adaptation) for efficient training and deployment on limited hardware.

🔄 Architecture Overview

Input Text → Embedding Layer → Qwen Layers (×28) → Attention (+LoRA) → MLP → LM Head → Output Text
📌 Components
🔹 Input Text

Raw input text is tokenized into numerical IDs using a tokenizer.

🔹 Embedding Layer

Converts token IDs into dense vector representations so the model can process them.

🔹 Qwen Transformer Layers (×28)

The backbone of the model consisting of 28 stacked transformer layers.
These layers capture contextual relationships using:

Self-attention
Feedforward networks
Residual connections
🔹 Attention + LoRA

The attention mechanism identifies relationships between tokens.

QLoRA is applied here to enable efficient fine-tuning:

Only small low-rank adapter weights are trained
Base model weights remain frozen
Significantly reduces memory usage
🔹 MLP (Feedforward Layer)

Further processes attention outputs to refine learned representations.

🔹 LM Head

Maps model outputs to vocabulary probabilities for next-token prediction.

🔹 Output Text

Generates the final response token-by-token.                                                                                                         What is PEFT?

PEFT (Parameter-Efficient Fine-Tuning) is a technique that avoids updating the full model weights.

Instead of training billions of parameters:

The base model is frozen
Only a small number of additional parameters are trained

👉 This drastically reduces:

Memory usage
Training time
Compute requirements
🔧 LoRA (Low-Rank Adaptation)

LoRA works by injecting small trainable matrices into existing layers.

Instead of modifying a weight matrix W, LoRA learns:

W
′
=W+ΔW=W+A⋅B

Where:

A∈R
d×r
B∈R
r×d
r≪d (low-rank)

👉 Only A and B are trained
👉 W remains frozen

📍 Where LoRA is applied

In this project, LoRA is applied to:

Attention layers (Query, Key, Value projections)

This is where most of the model’s learning capacity lies.

⚡ What is QLoRA?

QLoRA extends LoRA by combining it with quantization.

Key idea:
Load base model in 4-bit precision
Train LoRA adapters in higher precision
🔹 Benefits
🔻 Reduces GPU memory usage drastically
⚡ Enables training on GPUs like Tesla P100 (Kaggle)
💡 Maintains strong performance
🧠 BitsAndBytes (4-bit Quantization)

This project uses BitsAndBytes for efficient model loading.

Key features:
Loads model weights in 4-bit precision (NF4)
Reduces memory footprint by ~75%
Uses optimized CUDA kernels for fast computation
🔹 Quantization Details
Quantization Type: NF4 (Normal Float 4)
Compute dtype: float16
Double Quantization: optional (further compression)
📌 Why NF4?

NF4 is designed specifically for neural networks:
Better accuracy than standard int4
Preserves weight distribution more effectively .                                                                                                    In this project, QLoRA adapters are applied to:

q_proj (Query projection)
v_proj (Value projection)
Teach a large model new tasks without retraining the entire model.”
👉 These are critical parts of the attention mechanism, where most learning happens.                                                       
| Parameter        | Description                               |
| ---------------- | ----------------------------------------- |
| `r`              | Rank of LoRA matrices (controls capacity) |
| `lora_alpha`     | Scaling factor                            |
| `target_modules` | Layers where LoRA is applied              |
| `lora_dropout`   | Regularization                            |
| `bias`           | Whether bias is trained                   |  

Why QLoRA Adapters?
🔻 Train only ~1–5% of parameters
💾 Huge memory savings
⚡ Faster training
🧠 Maintains strong model performance
🖥️ Works on limited GPUs (e.g., GPU P100)

| Feature      | Benefit                      |
| ------------ | ---------------------------- |
| PEFT         | Reduces trainable parameters |
| LoRA         | Efficient adaptation         |
| QLoRA        | Enables low-memory training  |
| BitsAndBytes | Optimized quantization       |    
                                                                                                                                                   🔐 Hugging Face Authentication

This project uses a Hugging Face access token to download and load gated models such as Qwen from the Hugging Face Hub.

📌 Why is a token required?

Some models (e.g., Qwen) are:

🔒 Gated / restricted
Require user agreement before access

👉 A Hugging Face token ensures:

Authorized access
Secure model downloads
⚙️ Setup
1. Get your token
Go to: https://huggingface.co/settings/tokens
Create a token with read access
2. Authenticate in notebook
from huggingface_hub import login
login()

👉 Paste your token when prompted

3. Alternative (environment variable)
import os
os.environ["HF_TOKEN"] = "your_token_here"
⚠️ Security Note
❌ Do NOT hardcode your token in code or commit it to GitHub
❌ Do NOT share your token publicly
✅ Use environment variables or secret managers (e.g., Kaggle Secrets)
💡 Kaggle Setup

On Kaggle:

Go to Add-ons → Secrets
Add your Hugging Face token
Access it securely in code                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
## Benchmark Results

Environment:

* Platform: Kaggle Notebook
* GPU: Tesla P100-PCIE-16GB
* Model: Qwen2.5-1.5B
* Quantization: 4-bit (QLoRA)
* Quant Type: NF4
* Compute Dtype: FP16

Configuration:

* load_in_4bit: True
* bnb_4bit_quant_type: nf4
* bnb_4bit_compute_dtype: float16
* bnb_4bit_use_double_quant: False

Performance:

* Peak GPU Memory Usage: ~4.5 GB *(measured)*
* Training Time: XX minutes
* Batch Size: X
* Epochs: X

Observations:
QLoRA reduces memory usage significantly, allowing efficient fine-tuning of Qwen2.5-1.5B within ~5GB VRAM on a 16GB GPU.
