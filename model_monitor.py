#!/usr/bin/env python
"""
ML ENGINEERING: System Monitoring & Model Performance Tracking

Tracks CPU/Memory usage during model loading and inference.
Compares H5 vs ONNX performance for ML optimization.
"""

import psutil
import os
import time
from typing import Dict, List, Tuple

class ModelMonitor:
    """
    Monitor system resources during model loading and inference.
    Useful for ML optimization and deployment sizing.
    """
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.metrics = {}
        
    def get_memory_usage(self) -> float:
        """Returns current memory usage in MB"""
        return self.process.memory_info().rss / (1024 * 1024)
    
    def get_cpu_percent(self) -> float:
        """Returns current CPU usage as percentage"""
        return self.process.cpu_percent(interval=0.1)
    
    def start_tracking(self, event_name: str) -> None:
        """Start tracking an event"""
        self.metrics[event_name] = {
            'start_memory': self.get_memory_usage(),
            'start_time': time.time(),
            'start_cpu': self.get_cpu_percent(),
            'peak_memory': self.get_memory_usage(),
        }
    
    def end_tracking(self, event_name: str) -> Dict:
        """End tracking and return metrics"""
        if event_name not in self.metrics:
            return {}
        
        data = self.metrics[event_name]
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        data['end_time'] = end_time
        data['end_memory'] = end_memory
        data['elapsed_time'] = end_time - data['start_time']
        data['memory_increase'] = end_memory - data['start_memory']
        data['avg_cpu'] = self.get_cpu_percent()
        
        return data
    
    def print_report(self, event_name: str) -> None:
        """Print formatted performance report"""
        if event_name not in self.metrics:
            return
        
        data = self.metrics[event_name]
        if 'elapsed_time' not in data:
            return
        
        print(f"\n📊 Performance: {event_name}")
        print(f"  ⏱️  Time:       {data['elapsed_time']:.2f}s")
        print(f"  💾 Memory:     {data['start_memory']:.1f}MB → {data['end_memory']:.1f}MB (+{data['memory_increase']:.1f}MB)")
        print(f"  ⚙️  CPU:        {data['avg_cpu']:.1f}%")
    
    def compare_models(self, h5_metrics: Dict, onnx_metrics: Dict) -> None:
        """
        Compare H5 vs ONNX model performance
        Useful for understanding optimization gains
        """
        print("\n" + "="*70)
        print("🏆 H5 vs ONNX COMPARISON")
        print("="*70)
        
        if 'elapsed_time' in h5_metrics and 'elapsed_time' in onnx_metrics:
            h5_time = h5_metrics['elapsed_time']
            onnx_time = onnx_metrics['elapsed_time']
            speedup = (h5_time / onnx_time - 1) * 100
            
            print(f"\n⏱️  LOADING TIME:")
            print(f"   H5:   {h5_time:.2f}s")
            print(f"   ONNX: {onnx_time:.2f}s")
            print(f"   → {speedup:.1f}% faster" if speedup > 0 else f"   → {abs(speedup):.1f}% slower")
        
        if 'memory_increase' in h5_metrics and 'memory_increase' in onnx_metrics:
            h5_mem = h5_metrics['memory_increase']
            onnx_mem = onnx_metrics['memory_increase']
            savings = (h5_mem / onnx_mem - 1) * 100 if onnx_mem > 0 else 0
            
            print(f"\n💾 MEMORY USAGE:")
            print(f"   H5:   +{h5_mem:.1f}MB")
            print(f"   ONNX: +{onnx_mem:.1f}MB")
            print(f"   → {savings:.1f}% less memory" if savings > 0 else f"   → {abs(savings):.1f}% more memory")
        
        print("="*70 + "\n")

# Global monitor instance
monitor = ModelMonitor()


def print_system_status(title: str = "System Status") -> None:
    """Print current system resource usage"""
    memory_mb = psutil.virtual_memory().used / (1024 * 1024)
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    print(f"\n📊 {title}:")
    print(f"   System Memory:  {memory_mb:.0f}MB used")
    print(f"   System CPU:     {cpu_percent:.1f}%")
    print(f"   Process Memory: {monitor.get_memory_usage():.1f}MB")
    print(f"   Process CPU:    {monitor.get_cpu_percent():.1f}%\n")
