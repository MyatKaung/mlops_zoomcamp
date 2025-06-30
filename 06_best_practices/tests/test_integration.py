import os
import subprocess
import pandas as pd


def test_integration():
    """Integration test for the batch prediction pipeline"""
    
    # Set environment variables for S3 paths
    os.environ['INPUT_FILE_PATTERN'] = 's3://nyc-duration/in/{year:04d}/{month:02d}/test_data.parquet'
    os.environ['OUTPUT_FILE_PATTERN'] = 's3://nyc-duration/out/{year:04d}/{month:02d}/predictions.parquet'
    os.environ['AWS_ENDPOINT_URL'] = 'http://localhost:4566'
    
    # Run the batch script
    result = subprocess.run(
        ['python3', 'batch.py', '2023', '3'],
        capture_output=True,
        text=True,
        cwd='/Users/myatkaung/Desktop/MLOPS_Zoomcamp/06_best_practices'
    )
    
    # Check that the script ran successfully
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"
    
    # Download and verify the output
    output_file = 'predictions_test.parquet'
    download_result = subprocess.run([
        'aws', '--endpoint-url=http://localhost:4566', 's3', 'cp',
        's3://nyc-duration/out/2023/03/predictions.parquet',
        output_file
    ], capture_output=True, text=True)
    
    assert download_result.returncode == 0, f"Failed to download output: {download_result.stderr}"
    
    # Read and verify the predictions
    df_predictions = pd.read_parquet(output_file)
    
    # Verify the structure
    assert 'ride_id' in df_predictions.columns
    assert 'predicted_duration' in df_predictions.columns
    
    # Verify we have predictions for all input rows
    assert len(df_predictions) == 5  # We created 5 test records
    
    # Verify predictions are reasonable numbers (can be negative)
    assert all(df_predictions['predicted_duration'].notna())  # No NaN values
    assert len(df_predictions['predicted_duration']) > 0  # Has predictions
    
    # Clean up
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Calculate the sum of predicted durations as required by Q6
    sum_predicted_duration = df_predictions['predicted_duration'].sum()
    
    print(f"Integration test passed! Generated {len(df_predictions)} predictions.")
    print(f"Sum of predicted durations: {sum_predicted_duration:.2f} minutes")
    print(f"Mean predicted duration: {df_predictions['predicted_duration'].mean():.2f} minutes")