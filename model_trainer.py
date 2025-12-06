import torch
import os
from tqdm import tqdm
from lstm import StockLSTM
from transformer import StockTransformer
from stock_dataloader import create_stock_dataloader

def train_model(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, num_epochs: int=50, learning_rate: float=0.001, device: str='cpu') -> torch.nn.Module:
    """
    Train LSTM/Transformer model on stock data
    """

    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    model.train()

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(train_loader.dataset)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")

    return model

def save_model(model: torch.nn.Module, save_name: str, save_dir: str='models') -> str:
    """
    Save trained omdel + optimizer + metadata
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{save_name}")

    # Save everything needed for evaluation
    if 'LSTM' in model.__class__.__name__:
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': model.__class__.__name__,
            'input_size': getattr(model, 'input_size', 1),
            'hidden_size': getattr(model, 'hidden_size', 64),
            'num_layers': getattr(model, 'num_layers', 2),
            'dropout': getattr(model, 'dropout', 0.2),
            'architecture': 'StockLSTM'
            }, save_path)
    elif 'Transformer' in model.__class__.__name__:
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': model.__class__.__name__,
            'inp_dim': getattr(model, 'inp_dim', 1),
            'd_model': getattr(model, 'd_model', 64),
            'n_heads': getattr(model, 'n_heads', 4),
            'n_layers': getattr(model, 'n_layers', 3),
            'dim_feedforward': getattr(model, 'dim_feedforward', 256),
            'dropout': getattr(model, 'dropout', 0.1),
            'output_dim': getattr(model, 'output_dim', 1),
            'max_len': getattr(model, 'max_len', 500),
            'architecture': 'StockTransformer'
            }, save_path)

    print(f"✅ Model saved: {save_path}")
    return save_path

def load_model(load_path: str) -> torch.nn.Module:
    """
    Load trained model from disk
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"❌ Model file not found: {load_path}")
        
    checkpoint = torch.load(load_path, weights_only=True)
    model_type = checkpoint['model_type']

    if model_type == 'StockLSTM':
        model = StockLSTM(input_size=checkpoint['input_size'],
                          hidden_size=checkpoint['hidden_size'],
                          num_layers=checkpoint['num_layers'],
                          dropout=checkpoint['dropout'])
    elif model_type == 'StockTransformer':
        model = StockTransformer(inp_dim=checkpoint['inp_dim'],
                                 d_model=checkpoint['d_model'],
                                 n_heads=checkpoint['n_heads'],
                                 n_layers=checkpoint['n_layers'],
                                 dim_feedforward=checkpoint['dim_feedforward'],
                                 dropout=checkpoint['dropout'],
                                 output_dim=checkpoint['output_dim'],
                                 max_len=checkpoint['max_len'])
    else:
        raise ValueError(f"Unknown model type {model_type} in checkpoint.")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model loaded: {load_path}")
    return model


if __name__ == "__main__":
    ### Hyperparameters ###

    # Pick model: LSTM or Transformer
    model_choice = 'Transformer'   # Select 'LSTM' or 'Transformer'
    model_save_name = f'Stock{model_choice}_ModelMini'
    load_model = False

    # Dataloader
    SEQ_LEN = 100           # default 100; window for set of time-series data points
    BATCH_SIZE = 32         # default 32; increase if GPU mem allows
    STOCKS_PER_BUCKET = 5   # default 13; number of stocks per category bucket
    TRAIN_PER_BUCKET = 3    # default 10; number of training stocks per category bucket
    # LSTM-specific
    INPUT_SIZE = 1          # default 1; based on data
    HIDDEN_SIZE = 64        # default 64; analogous to D_MODEL; increase to 128 if underfitting
    NUM_LAYERS = 2          # default 2, re-evaluate if underfitting
    DROP_OUT = 0.2          # default 0.2; re-evaluate if overfitting
    # Transformer-specific
    INP_DIM = 1             # default 1; based on data
    D_MODEL = 64            # default 64; analogous to HIDDEN_SIZE; re-evaluate if underfitting
    N_HEADS = 4             # default 4; 64/4 = 16 - standard ratio
    N_LAYERS = 3            # default 3; re-evaluate if underfitting
    DIM_FEEDFORWARD = 256   # default 256; 4x D_MODEL is standard
    DROPOUT = 0.1           # default 0.1; re-evaluate if overfitting
    OUTPUT_DIM = 1          # default 1; based on data - next-day closing price
    MAX_LEN = 500           # default 500; should be > SEQ_LEN
    # Training
    NUM_EPOCHS = 50         # default 50; increase if underfitting
    LEARNING_RATE = 0.001   # default 0.001; drop to 3e-4 if unstable
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    stock_csv = 'selected_stocks_data.csv'  # Pre-downloaded stock prices
    metadata_csv = 'selected_stocks_quality.csv'  # Metadata with categories and qualities

    train_loader = create_stock_dataloader(stock_csv, metadata_csv, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
                                           stocks_per_bucket=STOCKS_PER_BUCKET, train_per_bucket=TRAIN_PER_BUCKET)['train_loader']

    if model_choice == 'LSTM':
        model = StockLSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROP_OUT).to(DEVICE)
    elif model_choice == 'Transformer':
        model = StockTransformer(inp_dim=INP_DIM, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                                 dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT, output_dim=OUTPUT_DIM, max_len=MAX_LEN).to(DEVICE)
    else:
        raise ValueError(f"Invalid model choice {model_choice}. Select 'LSTM' or 'Transformer'.")
    
    print(f"Training {model_choice} model on device: {DEVICE}")
    trained_model = train_model(model, train_loader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, device=DEVICE)
    save_model(trained_model, save_name=model_save_name)
