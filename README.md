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

precision  
recall  
F1 score  
false positive rate  
average inference latency  
p95 inference latency  

## Run

```bash
pip install -r requirements.txt
python main.py