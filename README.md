# Neural Shayari: Urdu Poetry Generation using Deep Learning 🖋️

This repository contains a complete deep learning NLP pipeline for generating Urdu poetry. It features the training of different neural network architectures (SimpleRNN, LSTM, and Custom Transformers) and extensive hyperparameter tuning to generate coherent and creative Urdu verses from specific seed words.

## 📖 Project Overview

The goal of this project is to model the linguistic patterns and structures of Urdu poetry using sequence-to-sequence deep learning models. By training on a corpus of Urdu verses, the models learn to predict the next token in a sequence, allowing them to compose original lines of poetry based on thematic seed words.

## 🗄️ Dataset & Preprocessing

The project utilizes the [ReySajju742/Urdu-Poetry-Dataset](https://huggingface.co/datasets/ReySajju742/Urdu-Poetry-Dataset) from Hugging Face.

**Preprocessing Pipeline:**
- **Text Cleaning:** Removed English characters, numbers, and special symbols.
- **Filtering:** Dropped verses with fewer than 5 characters to ensure content quality.
- **Tokenization:** Trained a custom tokenizer resulting in a vocabulary size of around 10,508 tokens.
- **Sequencing:** Generated N-gram token sequences, padded to a fixed `MAX_SEQ_LEN = 20`.
- **Data Splits:** 80% Training, 10% Validation, and 10% Testing.

## 🧠 Model Architectures

Three main baseline architectures were trained and evaluated using different optimizers (`Adam`, `RMSprop`, `SGD`):
1. **SimpleRNN:** (150 units, 2 hidden layers, 100-dim word embeddings, 0.2 dropout)
2. **LSTM:** (150 units, 2 hidden layers, 100-dim word embeddings, 0.2 dropout)
3. **Transformer:** A custom-built transformer leveraging `TokenAndPositionEmbedding` and `TransformerBlock` layers.

*The initial training phase is documented in `code/base models.ipynb`.*

## ⚙️ Hyperparameter Tuning

To optimize the generative capabilities, the best architecture-optimizer combinations (e.g., RNN+Adam, LSTM+RMSprop, Transformer+SGD) were isolated and tuned. We extensively tested **9+ different hyperparameter configurations for every model** to discover the absolute best setups. Models were evaluated using the **Perplexity** metric (derived from validation loss).

**Tuning Experiments (9+ configurations per model):**
- **Global Constraints:** Increased layers (3), higher Dropout (0.5), varied Learning Rate (0.0001), larger Batch Sizes (256), and extended Epochs (50).
- **Transformer-specific:** Increased Attention Heads (8), expanded Feed-Forward Dimensions (1024), and stacked more Transformer Blocks (4).

*The tuning experiments are documented in `code/hyperparameter.ipynb`.*

## 🎭 Text Generation

Poetry generation uses seed words as a starting point. To explore the trade-off between coherence and creativity, generation utilizes different **Temperatures**:
- `0.7`: Safer, more predictable structure.
- `1.0`: A balanced text output.
- `1.3`: Highly creative and diverse, prone to more experimental syntax.

**Seed Words Used:**
`محبت` (Love), `دل` (Heart), `شام` (Evening), `یاد` (Memory), `خوشی` (Happiness)

## 📁 Repository Structure

```text
├── code/
│   ├── base models.ipynb        # Data preprocessing, base models setup, and training
│   ├── hyperparameter.ipynb     # Hyperparameter isolation and targeted 
├── docs/
│   ├── FINAL_POETRY_RESULTS.csv # Model-generated verses from the baseline architectures
│   ├── TUNED_MODELS_POETRY.csv  # Model-generated verses from the specifically tuned models
│   └── report.pdf               # Comprehensive final project report
└── README.md
```

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Ahmednaveeed/Urdu-Poetry-Generator.git
   cd Urdu-Poetry-Generator
   ```
2. Make sure you have Jupyter Notebook or JupyterLab installed, along with standard data science and deep learning libraries (TensorFlow/Keras, Pandas, NumPy).
3. Open `code/base models.ipynb` to view the data pipeline and initial training. 
4. Check out the project report and generated verse metrics in the `docs/` folder.

## 📊 Results

Feel free to browse `docs/FINAL_POETRY_RESULTS.csv` and `docs/TUNED_MODELS_POETRY.csv` to read the generated combinations. The results highlight how shifting from simple recurrence (RNN/LSTM) to Attention-based mechanisms (Transformers), alongside hyperparameter tweaking, distinctly changes the rhythm and quality of the generated Urdu poetry.

> **Note:** While the results improved significantly through extensive hyperparameter tuning and architecture upgrades, it is important to acknowledge that the models could not achieve absolute perfection. This is primarily because the dataset used was relatively small, whereas deep learning sequence models typically require massive text corpora to truly master the deep structural nuances of poetry.
