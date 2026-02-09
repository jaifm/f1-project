import os
import fastf1
import pandas as pd
import numpy as np
import logging
from fastf1.core import Laps

# --- CONFIGURATION ---
CACHE_DIR = 'data/cache'  # Stores raw FastF1 downloads
OUTPUT_DIR = 'data/raw'   # Stores processed dataframes
YEARS = [2022, 2023, 2024]
# Set logging to see progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_directories():
    """Ensure data directories exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Enable FastF1 cache
    fastf1.Cache.enable_cache(CACHE_DIR)

def enrich_laps_with_weather(session):
    """
    Merges weather data (AirTemp, TrackTemp, Humidity) into the laps.
    FastF1 weather data is time-series; we map it to the start of each lap.
    """
    laps = session.laps
    weather = session.weather_data
    
    # Convert times to seconds for easier interpolation
    weather['TimeSec'] = weather['Time'].dt.total_seconds()
    laps['TimeSec'] = laps['Time'].dt.total_seconds()
    
    # Interpolate weather data to the exact time the lap finished
    # Note: Using 'Time' (end of lap) gives the conditions at the line. 
    # Ideally, want the average, but end-of-lap is a standard proxy.
    
    weather_cols = ['AirTemp', 'TrackTemp', 'Humidity', 'Rainfall']
    
    for col in weather_cols:
        # Interpolate weather data to match lap times
        laps[col] = np.interp(laps['TimeSec'], weather['TimeSec'], weather[col])
        
    return laps

def process_season(year):
    """Downloads and processes all races for a given year."""
    logging.info(f"Starting ingestion for Season {year}")
    
    # Get the schedule for the year
    schedule = fastf1.get_event_schedule(year)
    
    # Filter for official races only (exclude testing)
    races = schedule[schedule['EventFormat'] != 'testing']
    
    all_season_laps = []
    
    for _, race in races.iterrows():
        round_number = race['RoundNumber']
        event_name = race['EventName']
        
        logging.info(f"  -> Processing Round {round_number}: {event_name}")
        
        try:
            # Load the Race Session
            session = fastf1.get_session(year, round_number, 'R')
            session.load(telemetry=False, weather=True, messages=False) # Telemetry=False to save memory for now
            
            # 1. Filter: Valid Laps Only
            # pick_accurate() removes laps with safety cars, VSC, or erroneous timing
            # pick_wo_box() removes in-laps and out-laps
            clean_laps = session.laps.pick_accurate().pick_wo_box().reset_index(drop=True)
            
            if clean_laps.empty:
                logging.warning(f"     No clean laps found for {event_name}. Skipping.")
                continue

            # 2. Enrich: Add Weather Data
            clean_laps = enrich_laps_with_weather(session)
            
            # 3. Enrich: Add Context Metadata
            clean_laps['RoundNumber'] = round_number
            clean_laps['EventName'] = event_name
            clean_laps['Year'] = year
            clean_laps['CircuitLocation'] = race['Location']
            
            # 4. Filter: Keep only necessary columns to keep file size down
            # Will engineer complex features (fuel, traffic) in the next step
            keep_cols = [
                'Driver', 'LapTime', 'LapNumber', 'Stint', 'PitOutTime', 
                'PitInTime', 'Compound', 'TyreLife', 'FreshTyre',
                'Team', 'Year', 'RoundNumber', 'EventName', 'CircuitLocation',
                'AirTemp', 'TrackTemp', 'Humidity', 'Rainfall', 'TrackStatus'
            ]
            
            # Ensure only keep columns that actually exist
            available_cols = [c for c in keep_cols if c in clean_laps.columns]
            all_season_laps.append(clean_laps[available_cols])
            
        except Exception as e:
            logging.error(f"     Failed to process {event_name}: {e}")
            continue

    if not all_season_laps:
        return None

    # Combine all races into one DataFrame
    season_df = pd.concat(all_season_laps, ignore_index=True)
    
    # Convert LapTime to seconds (Float) immediately for easier math later
    season_df['LapTimeSeconds'] = season_df['LapTime'].dt.total_seconds()
    
    return season_df

if __name__ == "__main__":
    setup_directories()
    
    for year in YEARS:
        df = process_season(year)
        if df is not None:
            # Save as Parquet for efficient storage and faster loading later
            output_path = f"{OUTPUT_DIR}/f1_laps_{year}.parquet"
            df.to_parquet(output_path, index=False)
            logging.info(f"Saved {year} data to {output_path}")
            
    logging.info("Ingestion Complete.")