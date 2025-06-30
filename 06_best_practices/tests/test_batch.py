import pandas as pd
from datetime import datetime
import sys
import os

# Add the parent directory to the path so we can import batch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch import prepare_data


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    # Create test data
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]
    
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)
    
    categorical = ['PULocationID', 'DOLocationID']
    
    # Apply the prepare_data function
    actual_df = prepare_data(df, categorical)
    
    # Expected results:
    # Row 0: duration = 9 minutes (1:01 to 1:10), within range [1, 60]
    # Row 1: duration = 8 minutes (1:02 to 1:10), within range [1, 60]  
    # Row 2: duration = 59 seconds = ~0.98 minutes, outside range [1, 60]
    # Row 3: duration = 25 hours 1 minute = 1501 minutes, outside range [1, 60]
    
    # So we expect 2 rows in the result (rows 0 and 1)
    expected_data = [
        ('-1', '-1', dt(1, 1), dt(1, 10), 9.0),
        ('1', '1', dt(1, 2), dt(1, 10), 8.0),
    ]
    
    expected_columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration']
    expected_df = pd.DataFrame(expected_data, columns=expected_columns)
    
    # Check that we have the expected number of rows
    assert len(actual_df) == 2
    
    # Check the duration values
    assert actual_df.iloc[0]['duration'] == 9.0
    assert actual_df.iloc[1]['duration'] == 8.0
    
    # Check categorical columns are converted to strings
    assert actual_df.iloc[0]['PULocationID'] == '-1'
    assert actual_df.iloc[0]['DOLocationID'] == '-1'
    assert actual_df.iloc[1]['PULocationID'] == '1'
    assert actual_df.iloc[1]['DOLocationID'] == '1'
    
    print(f"Test passed! Expected 2 rows, got {len(actual_df)} rows.")