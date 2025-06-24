-- Create table for storing monitoring metrics
CREATE TABLE IF NOT EXISTS metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    date DATE NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    additional_info JSONB
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_metrics_date ON metrics(date);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);

-- Insert some sample data for testing
INSERT INTO metrics (date, metric_name, metric_value, additional_info) VALUES
('2024-03-01', 'fare_amount_quantile_0.5', 13.50, '{"drift_detected": true}'),
('2024-03-02', 'fare_amount_quantile_0.5', 14.00, '{"drift_detected": true}'),
('2024-03-03', 'fare_amount_quantile_0.5', 14.20, '{"drift_detected": true}');