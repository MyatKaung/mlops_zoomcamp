#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import os
import boto3


def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def read_data(filename, categorical):
    if filename.startswith('s3://'):
        # Configure S3 client for Localstack if endpoint URL is set
        s3_options = {}
        if 'AWS_ENDPOINT_URL' in os.environ:
            s3_options['endpoint_url'] = os.environ['AWS_ENDPOINT_URL']
        
        df = pd.read_parquet(filename, storage_options=s3_options)
    else:
        df = pd.read_parquet(filename)
    
    df = prepare_data(df, categorical)
    return df


def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 'taxi_type=yellow_year={year:04d}_month={month:02d}.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def save_data(df, output_file):
    if output_file.startswith('s3://'):
        # Configure S3 client for Localstack if endpoint URL is set
        s3_options = {}
        if 'AWS_ENDPOINT_URL' in os.environ:
            s3_options['endpoint_url'] = os.environ['AWS_ENDPOINT_URL']
        
        df.to_parquet(output_file, engine='pyarrow', index=False, storage_options=s3_options)
    else:
        df.to_parquet(output_file, engine='pyarrow', index=False)


def main(year, month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    
    categorical = ['PULocationID', 'DOLocationID']
    
    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    
    print('predicted mean duration:', y_pred.mean())
    
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    
    save_data(df_result, output_file)


if __name__ == '__main__':
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)