# Real-Time Sensor Anomaly Detection System

A real-time anomaly detection pipeline for multimodal sensor streams, designed for intrusion detection and perimeter monitoring scenarios.

The system models motion, audio, vibration, temperature, and door-state signals, then detects abnormal activity using classical ML and neural anomaly detection methods.

## Features

Real-time streaming inference loop  
Multimodal sensor data simulation  
Isolation Forest anomaly detection  
Autoencoder-based anomaly detection  
Noise, signal degradation, and partial sensor dropout simulation  
Precision, recall, false positive rate, and p95 latency benchmarking  

## Motivation

Physical security systems often need to detect unusual activity from noisy and incomplete sensor streams. This project simulates those conditions and evaluates anomaly detection models under real-time constraints.

## Sensor Inputs

motion  
audio  
vibration  
temperature  
door_open  

## Anomaly Scenarios

sudden motion spike  
audio spike  
vibration spike  
after-hours door opening  
sensor dropout  
combined motion/audio events  

## Models

Isolation Forest  
Autoencoder  

## Metrics

==================================================
Isolation Forest Metrics
==================================================
Metric                    |           Value
--------------------------------------------------
precision                 |          0.4988
recall                    |          0.7812
f1                        |          0.6088
false_positive_rate       |          0.0733
true_positives            |             200
false_positives           |             201
false_negatives           |              56
true_negatives            |            2543
avg_latency_ms            |          4.5678
p95_latency_ms            |          5.1832
max_latency_ms            |          8.4474
==================================================

==================================================
Autoencoder Metrics
==================================================
Metric                    |           Value
--------------------------------------------------
precision                 |          0.6649
recall                    |          1.0000
f1                        |          0.7988
false_positive_rate       |          0.0470
true_positives            |             256
false_positives           |             129
false_negatives           |               0
true_negatives            |            2615
avg_latency_ms            |          0.6422
p95_latency_ms            |          0.7888
max_latency_ms            |          1.4693
==================================================

## Run

```bash
pip install -r requirements.txt
python main.py