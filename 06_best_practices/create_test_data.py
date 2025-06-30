import pandas as pd
import os

def create_test_data():
    """Create test data for integration testing"""
    # Create a small dataset with the expected structure
    data = {
        'PULocationID': [1, 2, 3, 4, 5],
        'DOLocationID': [10, 20, 30, 40, 50],
        'tpep_pickup_datetime': [
            '2023-03-01 10:00:00',
            '2023-03-01 11:00:00', 
            '2023-03-01 12:00:00',
            '2023-03-01 13:00:00',
            '2023-03-01 14:00:00'
        ],
        'tpep_dropoff_datetime': [
            '2023-03-01 10:15:00',
            '2023-03-01 11:20:00',
            '2023-03-01 12:25:00', 
            '2023-03-01 13:30:00',
            '2023-03-01 14:35:00'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Convert datetime columns
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    
    # Save locally first
    output_file = 'test_data.parquet'
    df.to_parquet(output_file, index=False)
    
    # Upload to S3 using AWS CLI
    s3_path = 's3://nyc-duration/in/2023/03/test_data.parquet'
    os.system(f'aws --endpoint-url=http://localhost:4566 s3 cp {output_file} {s3_path}')
    
    print(f"Test data created and uploaded to {s3_path}")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

if __name__ == '__main__':
    create_test_data()