"""
Global Configuration for AI-Driven Smart Transportation & Logistics Optimization.

Contains all hyperparameters, file paths, and constants used across modules.
"""

import os

# ============================================================================
# Project Paths
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================================
# Data Generation Configuration
# ============================================================================
DATA_CONFIG = {
    "num_records": 50000,
    "start_date": "2016-01-01",
    "end_date": "2016-06-30",
    "random_seed": 42,
    # Real NYC Taxi Trip Duration data
    "real_train_file": os.path.join(PROJECT_ROOT, "nyc-taxi-trip-duration", "train.csv"),
    "real_test_file": os.path.join(PROJECT_ROOT, "nyc-taxi-trip-duration", "test.csv"),
    # Processed outputs
    "raw_data_file": os.path.join(DATA_DIR, "nyc_taxi_raw.csv"),
    "processed_data_file": os.path.join(DATA_DIR, "nyc_taxi_processed.csv"),
    "demand_data_file": os.path.join(DATA_DIR, "demand_aggregated.csv"),
}

# ============================================================================
# NYC Taxi Zone Configuration (Simplified — Top 50 Active Zones)
# ============================================================================
# Zone ID → (name, latitude, longitude)
NYC_ZONES = {
    4: ("Alphabet City", 40.7258, -73.9815),
    7: ("Astoria", 40.7644, -73.9235),
    12: ("Battery Park", 40.7033, -74.0170),
    13: ("Battery Park City", 40.7117, -74.0154),
    24: ("Bloomingdale", 40.7990, -73.9680),
    41: ("Central Harlem", 40.8116, -73.9465),
    42: ("Central Harlem North", 40.8194, -73.9420),
    43: ("Central Park", 40.7829, -73.9654),
    45: ("Chinatown", 40.7158, -73.9970),
    48: ("Clinton East", 40.7614, -73.9932),
    50: ("Clinton West", 40.7640, -73.9997),
    68: ("East Chelsea", 40.7427, -73.9945),
    74: ("East Harlem North", 40.7957, -73.9389),
    75: ("East Harlem South", 40.7903, -73.9440),
    79: ("East Village", 40.7265, -73.9815),
    87: ("Financial District North", 40.7081, -74.0090),
    88: ("Financial District South", 40.7023, -74.0131),
    90: ("Flatiron", 40.7395, -73.9903),
    100: ("Garment District", 40.7528, -73.9900),
    107: ("Gramercy", 40.7367, -73.9845),
    113: ("Greenwich Village North", 40.7340, -74.0003),
    114: ("Greenwich Village South", 40.7307, -74.0007),
    116: ("Hamilton Heights", 40.8253, -73.9501),
    120: ("Highbridge Park", 40.8440, -73.9340),
    125: ("Hudson Sq", 40.7267, -74.0076),
    127: ("Inwood", 40.8681, -73.9212),
    128: ("Inwood Hill Park", 40.8726, -73.9253),
    137: ("Kips Bay", 40.7424, -73.9801),
    140: ("Lenox Hill East", 40.7635, -73.9590),
    141: ("Lenox Hill West", 40.7685, -73.9620),
    142: ("Lincoln Square East", 40.7729, -73.9830),
    143: ("Lincoln Square West", 40.7748, -73.9866),
    144: ("Little Italy/NoLiTa", 40.7191, -73.9973),
    148: ("Lower East Side", 40.7150, -73.9843),
    151: ("Manhattan Valley", 40.7977, -73.9640),
    152: ("Manhattanville", 40.8152, -73.9553),
    153: ("Marble Hill", 40.8766, -73.9106),
    158: ("Meatpacking/West Village W", 40.7390, -74.0086),
    161: ("Midtown Center", 40.7549, -73.9840),
    162: ("Midtown East", 40.7527, -73.9720),
    163: ("Midtown North", 40.7610, -73.9780),
    164: ("Midtown South", 40.7491, -73.9887),
    166: ("Morningside Heights", 40.8075, -73.9626),
    170: ("Murray Hill", 40.7469, -73.9775),
    186: ("Penn Station/Madison Sq W", 40.7484, -73.9933),
    194: ("Randalls Island", 40.7935, -73.9211),
    202: ("Roosevelt Island", 40.7614, -73.9493),
    209: ("Seaport", 40.7069, -74.0037),
    211: ("SoHo", 40.7233, -73.9985),
    224: ("Stuy Town/PCV", 40.7317, -73.9771),
    229: ("Sutton Place/Turtle Bay N", 40.7577, -73.9645),
    230: ("Sutton Place/Turtle Bay S", 40.7530, -73.9680),
    231: ("Times Sq/Theatre District", 40.7580, -73.9855),
    232: ("TriBeCa/Civic Center", 40.7163, -74.0086),
    233: ("Two Bridges/Seward Park", 40.7127, -73.9879),
    234: ("UN/Turtle Bay South", 40.7488, -73.9700),
    236: ("Upper East Side North", 40.7736, -73.9566),
    237: ("Upper East Side South", 40.7652, -73.9600),
    238: ("Upper West Side North", 40.7937, -73.9700),
    239: ("Upper West Side South", 40.7810, -73.9760),
    243: ("Washington Heights North", 40.8550, -73.9350),
    244: ("Washington Heights South", 40.8400, -73.9400),
    246: ("West Chelsea/Hudson Yards", 40.7510, -74.0020),
    249: ("West Village", 40.7356, -74.0034),
    261: ("World Trade Center", 40.7118, -74.0131),
    262: ("Yorkville East", 40.7766, -73.9490),
    263: ("Yorkville West", 40.7790, -73.9550),
}

# High-demand zones (Manhattan core) — used for weighted data generation
HIGH_DEMAND_ZONES = [
    161, 162, 163, 164, 170, 186, 231, 237, 236, 239, 238,
    100, 48, 50, 68, 90, 107, 140, 141, 142, 143, 230, 229,
]

MEDIUM_DEMAND_ZONES = [
    43, 24, 151, 166, 41, 74, 75, 79, 113, 114, 144, 148,
    211, 232, 246, 249, 87, 88, 125, 224, 234,
]

# ============================================================================
# Demand Forecasting Model Configuration
# ============================================================================
FORECAST_CONFIG = {
    # Data parameters
    "lookback_window": 24,         # Hours of history as input
    "forecast_horizon": 6,         # Hours to predict ahead
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,

    # Transformer architecture
    "d_model": 64,                 # Embedding dimension
    "n_heads": 4,                  # Number of attention heads
    "n_encoder_layers": 4,         # Number of encoder layers
    "dim_feedforward": 256,        # FFN hidden dimension
    "dropout": 0.1,

    # Training parameters
    "batch_size": 64,
    "learning_rate": 1e-3,
    "max_epochs": 50,
    "early_stopping_patience": 8,
    "lr_scheduler_patience": 3,
    "lr_scheduler_factor": 0.5,

    # Paths
    "model_checkpoint": os.path.join(MODELS_DIR, "demand_transformer.pt"),
    "training_history": os.path.join(MODELS_DIR, "training_history.json"),
    "scaler_path": os.path.join(MODELS_DIR, "scaler.pkl"),
}

# ============================================================================
# Route Optimization Configuration
# ============================================================================
ROUTE_CONFIG = {
    "default_speed_kmh": 30,       # Average speed for time estimation
    "earth_radius_km": 6371.0,     # For Haversine distance
    "max_multi_stops": 10,         # Max stops for multi-stop optimization
    "two_opt_max_iterations": 1000,
}

# ============================================================================
# Evaluation Configuration
# ============================================================================
EVAL_CONFIG = {
    "forecast_sequence_lengths": [12, 24, 48],
    "forecast_model_dims": [32, 64, 128],
    "route_test_pairs": 20,        # Number of random source-dest pairs
    "results_dir": os.path.join(REPORTS_DIR, "results"),
}
os.makedirs(EVAL_CONFIG["results_dir"], exist_ok=True)
