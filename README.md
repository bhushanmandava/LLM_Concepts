# Transformer Model in PyTorch 

This repository contains a simple implementation of a Decoder-only Transformer model using PyTorch and Lightning. The model utilizes self-attention and position encoding to process sequential token data. The implementation is designed to predict the next token in a sequence, given an input sequence of token IDs.

## Structure of the Code

### 1. **Position Encoding**:  
   The `PositionEncoding` class computes the position encoding for token embeddings to introduce information about the order of tokens in the sequence. This is an essential part of the Transformer model to preserve sequential information.

### 2. **Attention Mechanism**:  
   The `Attention` class implements the scaled dot-product attention mechanism. It performs the following:
   - Queries, Keys, and Values (Q, K, V) are learned through linear layers.
   - The attention scores are computed using the similarity of the queries and keys, scaled by the square root of the dimension of the key vectors.
   - The attention scores are used to compute a weighted sum of the values, producing the final attention output.

### 3. **Decoder-only Transformer**:  
   The `DecoderOnlyTransformer` class is the core of the Transformer model:
   - It uses `Embedding` for token embeddings and `PositionEncoding` for positional encoding.
   - The `Attention` mechanism is applied to the token embeddings.
   - A residual connection is added to the output of the self-attention layer.
   - A fully connected layer is used to produce predictions.

### 4. **Training Loop**:  
   The model is trained using `LightningModule` and its `trainer.fit` method to handle the training loop and optimization.

### 5. **Input Data**:  
   The input data consists of tokenized sequences represented by integer IDs. The dataset is fed into the model to train the Transformer.

## How to Use

### Requirements

- PyTorch 1.9+
- PyTorch Lightning
- Python 3.x

You can install the required packages using pip:

```bash
pip install torch pytorch-lightning
```

### Example Usage

1. **Training**:

   To train the model, use the following command:

   ```python
   trainer = L.Trainer(max_epochs=30)
   trainer.fit(model, train_dataloaders=dataloader)
   ```

   This will train the model for 30 epochs using the data provided in the `dataloader`.

2. **Prediction**:

   You can use the model to make predictions after training by passing a sequence of token IDs into the model. The model will predict the next token in the sequence.

   ```python
   model_input = torch.tensor([token_to_id["what"],
                               token_to_id["is"],
                               token_to_id["transformer"],
                               token_to_id["<EOS>"]])
   predictions = model(model_input)
   predicted_id = torch.argmax(predictions[-1,:])
   print("Predicted Token:", id_to_token[predicted_id.item()])
   ```

## Components of the Model

### 1. **Embedding Layer**:
   This layer converts token IDs into continuous vector representations.

### 2. **Self-Attention Layer**:
   This layer computes the attention scores between tokens in the sequence and applies them to get the weighted sum of values.

### 3. **Residual Connection**:
   The output of the self-attention layer is added to the input embedding (residual connection), which helps improve training stability.

### 4. **Fully Connected Layer**:
   This layer projects the final embeddings back to the size of the token vocabulary and computes predictions.
