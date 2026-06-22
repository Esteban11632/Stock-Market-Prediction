def get_config():
    return {
        "ticker": "AAPL",
        "seq_length": 5,
        "num_lstm_layers": 4,
        "train_start_date": "2020-01-01",
        "train_end_date": "2023-12-31",
        "test_start_date": "2024-01-01",
        "batch_size": 64,
        "max_epochs": 190,
        "patience": 60,
        "wanted_features": [
        "Assets",
        "Liabilities",
        "NetIncomeLoss",
        "CashAndCashEquivalentsAtCarryingValue",
        "EarningsPerShareBasic",
        "CommonStockSharesOutstanding",
        "NetCashProvidedByUsedInOperatingActivities"
        ],
        "engineering_features": [
        "asset_growth_yoy",
        "liability_growth_yoy",
        "cash_growth_yoy",
        "net_income_growth_yoy",
        "eps_growth_yoy",
        "shares_growth_yoy",
        "operating_cf_growth_yoy"
        ]
    }