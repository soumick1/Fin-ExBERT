# FinExBERT: Financial Sentence Extraction with Graph-Augmented BERT

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org/)

> A state-of-the-art neural architecture for extracting relevant sentences from financial conversations using graph-augmented BERT with dependency parsing.

**Accepted at EMNLP 2025 Industry Track**

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## Overview

FinExBERT combines BERT's contextual understanding with graph neural networks to capture syntactic dependencies in financial conversations. The model achieves superior performance in extracting relevant sentences based on user intent, making it particularly effective for financial customer service applications.

### Problem Statement

Traditional sequence-to-sequence models struggle with:
- Complex financial terminology and context
- Long conversation dependencies
- Intent-based sentence extraction
- Domain-specific reasoning requirements

### Our Solution

FinExBERT addresses these challenges through:
- **Graph-Augmented Architecture**: Incorporates dependency parsing graphs to capture syntactic relationships
- **Financial Domain Adaptation**: LoRA fine-tuning on financial datasets
- **Intent-Aware Extraction**: Semantic similarity matching for targeted sentence selection
- **Efficient Training**: Mixed precision training with gradient accumulation

## Key Features

- 🏆 **State-of-the-art Performance**: Outperforms baseline BERT by 37% in accuracy on financial conversation tasks
- 🧠 **Graph Neural Networks**: Integrates dependency parsing for enhanced linguistic understanding
- 💰 **Financial Domain Expertise**: Pre-trained on financial conversation data
- ⚡ **Production Ready**: Optimized for real-world deployment with batched inference
- 🔧 **Flexible Architecture**: Configurable model components for different use cases
- 📊 **Comprehensive Evaluation**: Extensive ablation studies and evaluation metrics

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch 1.9 or higher
- CUDA 11.0+ (for GPU acceleration)


### Install from Source

```bash
git clone https://github.com/soumick1/Fin-ExBERT.git
cd finexbert
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/yourusername/finexbert.git
cd finexbert
pip install -e ".[dev]"
```

## Quick Start

### Download the model weights



### Basic Usage

```python
from finexbert import FinExBERTPredictor

# Initialize the model
predictor = FinExBERTPredictor.from_pretrained("finexbert-base")

# Extract relevant sentences
transcript = """
Customer: I'm interested in opening a savings account.
Agent: Great! Our current rate is 2.5% APY.
Customer: What's the minimum balance required?
Agent: The minimum balance is $500.
"""

intent = "customer asks about account requirements"
relevant_sentences = predictor.extract_sentences(transcript, intent)

for sentence, score in relevant_sentences:
    print(f"Score: {score:.3f} | {sentence}")
```

### Advanced Configuration

```python
from finexbert import FinExBERTConfig, FinExBERTPredictor

# Custom configuration
config = FinExBERTConfig(
    model=ModelConfig(
        model_name="bert-large-uncased",
        gnn_dim=256,
        dropout_prob=0.2
    ),
    training=TrainingConfig(
        batch_size=32,
        learning_rate=1e-5
    )
)

predictor = FinExBERTPredictor(config=config)
```

## Model Architecture

![FinExBERT Architecture](docs/images/architecture.png)

### Core Components

1. **BERT Encoder**: Contextual embeddings for input sequences
2. **Dependency Graph Parser**: SpaCy-based syntactic analysis
3. **Graph Neural Network**: Message passing over dependency graphs
4. **Fusion Layer**: Combines BERT and GNN representations
5. **Classification Head**: Intent-aware sentence scoring

### Technical Details

- **Base Model**: BERT-base-uncased (110M parameters)
- **GNN Architecture**: Simple message passing with attention
- **Training Strategy**: LoRA adaptation + full fine-tuning
- **Optimization**: AdamW with linear warmup and decay

## Training

### Prepare Your Data

```python
from finexbert.data import SentenceDataset

# Format: Excel file with columns 'Claude_Call' and 'Sel_K'
dataset = SentenceDataset("your_data.xlsx")
```

### Train the Model

```python
from finexbert import FinExBERTTrainer

trainer = FinExBERTTrainer(
    model=model,
    config=config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```

### Custom Training Script

See `examples/train_model.py` for a complete training example.

## Evaluation

### Ablation Studies

We provide comprehensive ablation studies comparing:

- Baseline BERT vs. Graph-Augmented BERT
- Different GNN architectures
- Various training strategies
- Domain adaptation techniques

```python
from finexbert.evaluation import run_ablation_study

results = run_ablation_study(
    models=["bert-baseline", "finexbert"],
    dataset="financial-conversations",
    metrics=["accuracy", "f1", "precision", "recall"]
)
```

### Performance Metrics

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| BERT Baseline | 0.323 | 0.163 | 0.145 | 0.189 |
| FinExBERT | 0.694 | 0.418 | 0.456 | 0.391 |
| **Improvement** | **+37%** | **+26%** | **+31%** | **+20%** |

## API Reference

### Core Classes

#### `FinExBERTPredictor`

Main inference class for sentence extraction.

```python
class FinExBERTPredictor:
    def __init__(self, model_path: str, config: FinExBERTConfig = None)
    def extract_sentences(self, text: str, intent: str, **kwargs) -> List[Tuple[str, float]]
    def batch_extract(self, texts: List[str], intents: List[str]) -> List[List[Tuple[str, float]]]
    @classmethod
    def from_pretrained(cls, model_name: str) -> 'FinExBERTPredictor'
```

#### `FinExBERTTrainer`

Training utilities for model fine-tuning.

```python
class FinExBERTTrainer:
    def __init__(self, model, config, train_dataset, eval_dataset)
    def train(self) -> Dict[str, float]
    def evaluate(self) -> Dict[str, float]
    def save_model(self, path: str)
```

### Configuration

All model and training parameters are configurable through the `FinExBERTConfig` class.

## Examples

### 1. Financial Intent Extraction

```python
# Extract sentences related to specific financial topics
predictor = FinExBERTPredictor.from_pretrained("finexbert-base")

transcript = load_conversation("customer_call.txt")
intents = [
    "customer asks about loan rates",
    "agent explains fees",
    "customer requests account information"
]

for intent in intents:
    sentences = predictor.extract_sentences(transcript, intent, top_k=3)
    print(f"\n{intent}:")
    for sentence, score in sentences:
        print(f"  {score:.3f}: {sentence}")
```

### 2. Batch Processing

```python
# Process multiple conversations efficiently
conversations = load_conversations("data/conversations.jsonl")
intents = ["customer complaint", "product inquiry", "account issue"]

results = predictor.batch_extract(
    texts=[conv["transcript"] for conv in conversations],
    intents=intents * len(conversations)
)
```

### 3. Custom Model Training

See `examples/` directory for complete training scripts:

- `train_model.py`: Full model training pipeline
- `evaluate_model.py`: Comprehensive evaluation
- `ablation_study.py`: Ablation study reproduction

## Citation

If you use FinExBERT in your research, please cite:

```bibtex
@inproceedings{finexbert2024,
  title={FinExBERT: Financial Sentence Extraction with Graph-Augmented BERT},
  author={Your Name and Co-authors},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track},
  year={2024},
  publisher={Association for Computational Linguistics}
}
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/yourusername/finexbert.git
cd finexbert
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of [Transformers](https://github.com/huggingface/transformers) by Hugging Face
- Graph processing with [SpaCy](https://spacy.io/)
- Training infrastructure powered by [PyTorch](https://pytorch.org/)

## Support

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/finexbert/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/finexbert/discussions)

---

<div align="center">
  <strong>FinExBERT</strong> - Advancing Financial NLP with Graph-Augmented Models
</div>
