#!/usr/bin/env python

import pandas as pd
import numpy as np
import pickle
import os
import sys

def load_model():
    """Load the pre-trained model and vectorizer from the Docker image."""
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return model, dv

def read_data(filename):
    """Read data from parquet file."""
    df = pd.read_parquet(filename)
    return df

def prepare_features(df, categorical, dv):
    """Prepare features for prediction."""
    # Compute duration
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    
    # Filter outliers
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)].copy()
    
    # Prepare categorical features
    df[categorical] = df[categorical].astype(str)
    
    # Transform features
    dicts = df[categorical].to_dict(orient='records')
    X = dv.transform(dicts)
    
    return X, df

def save_results(df, y_pred, year, month, output_file):
    """Save prediction results to parquet file."""
    # Create ride_id column
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    # Create results dataframe
    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })
    
    # Save as parquet
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    
    return df_result

def run(year, month):
    """Main function to run the scoring pipeline."""
    
    # Load pre-trained model
    print('Loading pre-trained model...')
    model, dv = load_model()
    print('Model loaded successfully')
    
    # Load data
    print(f'Loading data for {year}-{month:02d}...')
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = read_data(input_file)
    print(f'Data loaded: {len(df)} records')
    
    # Prepare features
    print('Preparing features...')
    categorical = ['PULocationID', 'DOLocationID']
    X, df_processed = prepare_features(df, categorical, dv)
    print(f'Data after filtering: {len(df_processed)} records')
    
    # Make predictions
    print('Making predictions...')
    y_pred = model.predict(X)
    print(f'Predictions made for {len(y_pred)} records')
    
    # Calculate standard deviation and mean
    std_pred = np.std(y_pred)
    mean_pred = np.mean(y_pred)
    print(f'Standard deviation of predicted duration: {std_pred:.2f}')
    print(f'Mean predicted duration: {mean_pred:.2f}')
    
    # Save results
    output_file = f'predictions_{year:04d}_{month:02d}.parquet'
    print(f'Saving results to {output_file}...')
    df_result = save_results(df_processed, y_pred, year, month, output_file)
    
    # Check file size
    file_size = os.path.getsize(output_file)
    file_size_mb = file_size / (1024 * 1024)
    print(f'File size: {file_size_mb:.0f}M')
    
    print('Scoring completed successfully!')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python score_docker.py <year> <month>")
        print("Example: python score_docker.py 2023 5")
        sys.exit(1)
    
    try:
        year = int(sys.argv[1])
        month = int(sys.argv[2])
        
        if month < 1 or month > 12:
            print("Error: Month must be between 1 and 12")
            sys.exit(1)
            
        run(year, month)
    except ValueError:
        print("Error: Year and month must be integers")
        sys.exit(1)