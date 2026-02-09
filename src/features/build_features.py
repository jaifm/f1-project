import pandas as pd
import numpy as np
import logging
import os

# --- CONFIGURATION ---
RAW_DATA_PATH = 'data/raw'
PROCESSED_DATA_PATH = 'data/processed'
OUTPUT_FILE = 'f1_training_data.parquet'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class FeatureEngineer:
    def __init__(self):
        self.df = None

    def load_raw_data(self):
        """Loads and merges all yearly parquet files."""
        files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith('.parquet')]
        if not files:
            raise FileNotFoundError(f"No data found in {RAW_DATA_PATH}. Run ingest_data.py first.")
        
        dfs = []
        for file in files:
            path = os.path.join(RAW_DATA_PATH, file)
            dfs.append(pd.read_parquet(path))
            logging.info(f"Loaded {file}")
            
        self.df = pd.concat(dfs, ignore_index=True)
        logging.info(f"Total raw samples: {len(self.df)}")

    def calculate_fuel_mass(self):
        """
        Estimates fuel mass.
        Rule of thumb: F1 cars start with ~110kg and burn ~1.7kg per lap.
        """
        # Calculate max laps for each race to estimate fuel load at start
        # (This handles different race lengths like Spa vs Monaco)
        race_lengths = self.df.groupby(['Year', 'EventName'])['LapNumber'].max().to_dict()
        
        def get_fuel(row):
            total_laps = race_lengths.get((row['Year'], row['EventName']), 55) # Default to 55 if missing
            laps_remaining = total_laps - row['LapNumber']
            # Fuel = LapsRemaining * BurnRate + SafetyMargin(5kg)
            return (laps_remaining * 1.7) + 5

        self.df['FuelMass'] = self.df.apply(get_fuel, axis=1)
        # Clip negative fuel (in case of extra laps/errors) to min 2kg
        self.df['FuelMass'] = self.df['FuelMass'].clip(lower=2.0)
        logging.info("Feature Engineered: FuelMass")

    def encode_physics_features(self):
        """
        Transforms categorical compounds and cleans numeric features.
        """
        # 1. Compound Softness Scale (1=Hard, 2=Medium, 3=Soft)
        # This helps the model learn that Soft > Medium > Hard in terms of grip
        compound_map = {'HARD': 1, 'MEDIUM': 2, 'SOFT': 3, 'INTERMEDIATE': 0, 'WET': 0}
        self.df['Compound_Softness'] = self.df['Compound'].map(compound_map)
        
        # Filter out Wet/Inter races for the "Dry" Physics Model
        initial_count = len(self.df)
        self.df = self.df[self.df['Compound_Softness'] > 0]
        logging.info(f"Filtered Wet/Inter laps. Dropped {initial_count - len(self.df)} laps.")

        # 2. Tyre Life (Cast to integer)
        self.df['TyreAge'] = self.df['TyreLife'].fillna(1).astype(int)

        # 3. Track Temp (Fill missing with median of that race)
        self.df['TrackTemp'] = self.df.groupby(['Year', 'EventName'])['TrackTemp'] \
                                      .transform(lambda x: x.fillna(x.median()))
        
        logging.info("Feature Engineered: Compound_Softness, TyreAge, TrackTemp")

    def create_target_variable(self):
        """
        Cleans the target variable (LapTime).
        Remove extreme outliers (e.g., > 105% of median pace) that aren't picked up by FastF1 filters.
        """
        # Calculate Median Lap Time per Race
        race_medians = self.df.groupby(['Year', 'EventName'])['LapTimeSeconds'].median()
        
        def is_outlier(row):
            median_pace = race_medians.loc[(row['Year'], row['EventName'])]
            # If lap is 20% slower than median (e.g. spin, damage), drop it.
            # Trying to model "Tire Wear", not "Crashes".
            return row['LapTimeSeconds'] > (median_pace * 1.20)

        # Apply filter
        outliers = self.df.apply(is_outlier, axis=1)
        self.df = self.df[~outliers]
        
        logging.info(f"Removed {outliers.sum()} slow outliers (spins/damage).")

    def save_processed_data(self):
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        path = os.path.join(PROCESSED_DATA_PATH, OUTPUT_FILE)
        self.df.to_parquet(path, index=False)
        logging.info(f"Saved processed data to {path} ({len(self.df)} rows)")

if __name__ == "__main__":
    fe = FeatureEngineer()
    fe.load_raw_data()
    fe.calculate_fuel_mass()
    fe.encode_physics_features()
    fe.create_target_variable()
    fe.save_processed_data()