#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import pickle
import os

def load_model():
    """Load and train a simple model for demonstration purposes.
    In practice, you would load a pre-trained model."""
    
    # Load training data to create model
    df_jan = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet')
    
    # Compute duration
    df_jan['duration'] = (df_jan['tpep_dropoff_datetime'] - df_jan['tpep_pickup_datetime']).dt.total_seconds() / 60
    
    # Filter outliers
    df_jan = df_jan[(df_jan['duration'] >= 1) & (df_jan['duration'] <= 60)].copy()
    
    # Prepare features
    categorical = ['PULocationID', 'DOLocationID']
    df_jan[categorical] = df_jan[categorical].astype(str)
    
    train_dicts = df_jan[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df_jan['duration'].values
    
    # Train model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    return lr, dv

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

def run():
    """Main function to run the scoring pipeline."""
    # Parameters
    year = 2023
    month = 3
    
    # Load model
    print('Loading model...')
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
    
    # Calculate standard deviation
    std_pred = np.std(y_pred)
    print(f'Standard deviation of predicted duration: {std_pred:.2f}')
    
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
    run()