def get_config():
    return {
        "ticker": "VOO",
        "seq_length": 5,
        "num_lstm_layers": 4,
        "train_start_date": "2020-01-01",
        "train_end_date": "2023-12-31",
        "test_start_date": "2024-01-01",
        "batch_size": 64,
        "max_epochs": 190,
        "patience": 60
    }