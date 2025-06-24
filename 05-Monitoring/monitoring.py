import pandas as pd
import psycopg2
import json
from datetime import datetime
from evidently.report import Report
from evidently.metrics import ColumnQuantileMetric, ColumnDriftMetric

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="monitoring",
        user="postgres",
        password="password",
        port="5433"
    )

def store_metric(date, metric_name, metric_value, additional_info=None):
    """Store metric in PostgreSQL database"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute(
        "INSERT INTO metrics (date, metric_name, metric_value, additional_info) VALUES (%s, %s, %s, %s)",
        (date, metric_name, metric_value, json.dumps(additional_info) if additional_info else None)
    )
    
    conn.commit()
    cur.close()
    conn.close()

# Load the March 2024 data
df = pd.read_parquet('data/green_tripdata_2024-03.parquet')

# Convert lpep_pickup_datetime to datetime objects
df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])

# Sort by pickup time
df = df.sort_values('lpep_pickup_datetime')

# Get the range of dates in the dataset
date_range = pd.date_range(df['lpep_pickup_datetime'].min().date(), df['lpep_pickup_datetime'].max().date())

max_quantile = 0
max_date = None

# Use first day as reference data for drift detection
first_day = date_range[0]
reference_data = df[df['lpep_pickup_datetime'].dt.date == first_day.date()]

print("Monitoring data quality metrics for March 2024 Green Taxi data")
print("Metrics: ColumnQuantileMetric (quantile=0.5) and ColumnDriftMetric for fare_amount")
print("="*80)

# Iterate through each day in March
for day in date_range:
    # Filter data for the current day
    current_day_data = df[df['lpep_pickup_datetime'].dt.date == day.date()]

    if not current_day_data.empty and len(current_day_data) > 10:  # Ensure sufficient data
        # Create a report with both metrics
        report = Report(metrics=[
            ColumnQuantileMetric(column_name="fare_amount", quantile=0.5),
            ColumnDriftMetric(column_name="fare_amount")
        ])

        # Run the report on the current day's data
        report.run(reference_data=reference_data, current_data=current_day_data)

        # Get the results
        result = report.as_dict()
        quantile_value = result['metrics'][0]['result']['current']['value']
        drift_detected = result['metrics'][1]['result']['drift_detected']
        
        print(f"Date: {day.date()}, Quantile 0.5: {quantile_value:.2f}, Drift detected: {drift_detected}")
        
        # Store metrics in database
        try:
            store_metric(
                date=day.date(),
                metric_name="fare_amount_quantile_0.5",
                metric_value=quantile_value,
                additional_info={"drift_detected": drift_detected}
            )
            print(f"  ✓ Stored metrics for {day.date()} in database")
        except Exception as e:
            print(f"  ✗ Failed to store metrics for {day.date()}: {e}")
        
        if quantile_value > max_quantile:
            max_quantile = quantile_value
            max_date = day.date()

print("="*70)
print(f"Maximum value of quantile = 0.5 on the 'fare_amount' column: {max_quantile:.2f}")
print(f"Date with maximum quantile: {max_date}")
print(f"\nChosen additional metric: ColumnDriftMetric")
print("This metric detects statistical drift in the fare_amount column by comparing")
print("each day's distribution against the reference day (first day of March).")