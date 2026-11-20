| Variant | Size (MB) | Test accuracy | Macro-F1 | Latency p50 (ms) | Latency p95 (ms) |
|---|---|---|---|---|---|
| keras | 21.90 | 0.852 | 0.822 | 156.6 | 220.5 |
| tflite_fp32 | 8.90 | 0.852 | 0.822 | 8.6 | 13.1 |
| tflite_fp16 | 4.52 | 0.852 | 0.822 | 8.5 | 12.9 |
| tflite_int8_dynamic | 2.53 | 0.852 | 0.822 | 20.7 | 24.1 |

Single-image CPU latency, 1 thread, n_test=27.
