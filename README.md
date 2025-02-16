# NLP_practice_IMDB

## Repository Overview
This repository contains NLP practice projects using PyTorch, focusing on sentiment analysis with the IMDB dataset. Two pre-trained models from Hugging Face are used:
- **bert-base-uncased**
- **ModernBERT-base**

## Configuration Files (`configs`)
Each `.yaml` file contains:
- The model name to load from Hugging Face.
- Experimental settings, such as:
  - Validation and test dataset size ratios
  - Batch size and number of epochs
  - Other training hyperparameters

## Code Structure
### `data.py`
- Loads the IMDB sentiment analysis dataset.
- Combines train and test sets, then splits into training, validation, and test sets (9:1:1).
- Token type IDs are included only when using **bert-base-uncased**, as **ModernBERT-base** does not require them.

### `model.py`
- Adds a classification head atop a pre-trained encoder for sentiment prediction.
- Outputs logits (0 or 1) using encoder representations.
- Uses **CrossEntropy** loss for optimization.

### `utils.py`
- Loads the selected model’s configuration file.

### `main.py`
Workflow summary:
1. **Set Device:** Use `'cuda'` if available.
2. **Load Data:** Invoke `get_dataloader()` from `data.py`.
3. **Set Optimizer:** Use the Adam optimizer.
4. **Initialize WandB:** Log into Weights and Biases (WandB) and set up the project directory.
5. **Train the Model:**
   - Train for 5 epochs.
   - Validate at each epoch.
   - Save checkpoints after each epoch.
6. **Test the Model:** Evaluate the best checkpoint on the test set.

## Results
Performance on **WandB**:
| **Model Name**        | **Test Loss** | **Test Accuracy** |
|-----------------------|--------------|-------------------|
| `bert-base-uncased`   | 0.0926       | 0.9778            |
| `ModernBERT-base`     | 0.0746       | 0.9832            |

## Discussion
As specified above, 'ModernBERT-base' outperformed 'bert-base-uncased'. Why so?
- Architectural Improvements: Disabled bias terms, RoPE, Pre-normalization, GeGLU activation.
- Training techniques: OLMo tokenizer, sequence packing, batch size scheduling, context length extension.

Conducting NLP experiments requires iterative testing with various configurations. Key best practices include:
- **Modularization:** Organize code into reusable components for flexible experimentation.
- **Documentation:** Maintain clear annotations for collaborative development.
- **Refactoring:** Keep code concise and maintainable.

This repository exemplifies best practices for model configuration, experiment tracking, and modular code design for NLP tasks.