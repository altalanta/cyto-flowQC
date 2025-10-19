"""Real-time processing and streaming analysis for cytoflow-qc."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# Optional dependencies for real-time processing
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False


class WebSocketProcessor:
    """WebSocket-based real-time data processor."""

    def __init__(self, ws_url: str, buffer_size: int = 10000):
        """Initialize WebSocket processor.

        Args:
            ws_url: WebSocket URL for data source
            buffer_size: Size of data buffer
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets library required for real-time processing")

        self.ws_url = ws_url
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self._running = False
        self._processing_thread = None
        self._data_handlers = []

    def add_data_handler(self, handler: Callable[[pd.DataFrame], None]) -> None:
        """Add a data processing handler.

        Args:
            handler: Function to call with incoming data
        """
        self._data_handlers.append(handler)

    def start_processing(self) -> None:
        """Start real-time data processing."""
        if self._running:
            return

        self._running = True

        async def process_websocket():
            try:
                async with websockets.connect(self.ws_url) as websocket:
                    print(f"🔗 Connected to WebSocket: {self.ws_url}")

                    while self._running:
                        try:
                            # Receive data
                            message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            data = json.loads(message)

                            # Convert to DataFrame
                            if isinstance(data, dict):
                                df = pd.DataFrame([data])
                            elif isinstance(data, list):
                                df = pd.DataFrame(data)
                            else:
                                continue

                            # Add to buffer
                            self.buffer.append(df)

                            # Process with handlers
                            for handler in self._data_handlers:
                                try:
                                    handler(df)
                                except Exception as e:
                                    print(f"Error in data handler: {e}")

                        except asyncio.TimeoutError:
                            continue
                        except websockets.exceptions.ConnectionClosed:
                            print("🔌 WebSocket connection closed")
                            break
                        except Exception as e:
                            print(f"Error processing WebSocket data: {e}")
                            break

            except Exception as e:
                print(f"WebSocket connection error: {e}")
            finally:
                self._running = False

        # Start async processing in thread
        self._processing_thread = threading.Thread(
            target=lambda: asyncio.run(process_websocket()),
            daemon=True
        )
        self._processing_thread.start()

        print("🚀 Real-time processing started")

    def stop_processing(self) -> None:
        """Stop real-time data processing."""
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)
        print("🛑 Real-time processing stopped")

    def get_buffer_data(self) -> pd.DataFrame:
        """Get all data currently in buffer."""
        if not self.buffer:
            return pd.DataFrame()

        return pd.concat(list(self.buffer), ignore_index=True)

    def clear_buffer(self) -> None:
        """Clear the data buffer."""
        self.buffer.clear()


class StreamingProcessor:
    """Streaming processor for continuous data analysis."""

    def __init__(self, output_dir: str | Path, window_size: int = 1000):
        """Initialize streaming processor.

        Args:
            output_dir: Directory for output files
            window_size: Size of rolling window for analysis
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        self._rolling_stats = {}
        self._data_handlers = []

    def add_analysis_handler(self, handler: Callable[[pd.DataFrame], dict[str, Any]]) -> None:
        """Add an analysis handler.

        Args:
            handler: Function that takes DataFrame and returns analysis results
        """
        self._data_handlers.append(handler)

    def process_streaming_data(self, new_data: pd.DataFrame) -> dict[str, Any]:
        """Process new streaming data.

        Args:
            new_data: New data chunk

        Returns:
            Dictionary of analysis results
        """
        results = {}

        # Update rolling statistics
        self._update_rolling_stats(new_data)

        # Run analysis handlers
        for handler in self._data_handlers:
            try:
                handler_results = handler(new_data)
                results.update(handler_results)
            except Exception as e:
                print(f"Error in analysis handler: {e}")

        # Save results if significant
        if results:
            timestamp = pd.Timestamp.now()
            results_file = self.output_dir / f"streaming_results_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"

            with open(results_file, 'w') as f:
                json.dump({
                    "timestamp": timestamp.isoformat(),
                    "results": results,
                    "data_points": len(new_data)
                }, f, indent=2)

        return results

    def _update_rolling_stats(self, data: pd.DataFrame) -> None:
        """Update rolling statistics for data quality monitoring."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col not in self._rolling_stats:
                self._rolling_stats[col] = deque(maxlen=self.window_size)

            self._rolling_stats[col].extend(data[col].dropna())

    def get_rolling_stats(self) -> dict[str, dict[str, float]]:
        """Get current rolling statistics."""
        stats = {}

        for col, values in self._rolling_stats.items():
            if values:
                values_array = np.array(values)
                stats[col] = {
                    "mean": float(values_array.mean()),
                    "std": float(values_array.std()),
                    "median": float(np.median(values_array)),
                    "count": len(values),
                    "min": float(values_array.min()),
                    "max": float(values_array.max())
                }

        return stats


class RealTimeMonitor:
    """Real-time monitoring dashboard for streaming data."""

    def __init__(self, port: int = 8081):
        """Initialize real-time monitor.

        Args:
            port: Port for monitoring dashboard
        """
        self.port = port
        self._monitoring_data = {}
        self._alerts = []

    def update_monitoring_data(self, data: dict[str, Any]) -> None:
        """Update monitoring data.

        Args:
            data: Dictionary of monitoring metrics
        """
        self._monitoring_data.update(data)
        self._monitoring_data["timestamp"] = pd.Timestamp.now().isoformat()

    def add_alert(self, alert_type: str, message: str, severity: str = "info") -> None:
        """Add an alert.

        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity (info, warning, error)
        """
        alert = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": pd.Timestamp.now().isoformat()
        }

        self._alerts.append(alert)

        # Keep only recent alerts
        if len(self._alerts) > 100:
            self._alerts = self._alerts[-100:]

    def get_monitoring_dashboard_data(self) -> dict[str, Any]:
        """Get data for monitoring dashboard."""
        return {
            "monitoring": self._monitoring_data,
            "alerts": self._alerts[-10:],  # Last 10 alerts
            "alert_count": len([a for a in self._alerts if a["severity"] in ["warning", "error"]])
        }


class DataStreamSimulator:
    """Simulator for testing real-time processing."""

    def __init__(self, output_file: str | Path, events_per_second: int = 100):
        """Initialize data stream simulator.

        Args:
            output_file: File to write simulated data
            events_per_second: Rate of data generation
        """
        self.output_file = Path(output_file)
        self.events_per_second = events_per_second
        self._running = False

    def start_simulation(self, duration_seconds: int = 60) -> None:
        """Start data simulation.

        Args:
            duration_seconds: How long to run simulation
        """
        self._running = True

        def simulate():
            start_time = time.time()

            with open(self.output_file, 'w') as f:
                # Write header
                header = "FSC-A,FSC-H,SSC-A,CD3-A,CD19-A,CD56-A\n"
                f.write(header)

                while self._running and (time.time() - start_time) < duration_seconds:
                    # Generate synthetic FCS data
                    n_events = self.events_per_second

                    data = []
                    for _ in range(n_events):
                        event = {
                            "FSC-A": np.random.lognormal(6, 0.5),
                            "FSC-H": np.random.lognormal(6, 0.5),
                            "SSC-A": np.random.lognormal(5, 0.5),
                            "CD3-A": np.random.lognormal(4, 0.8),
                            "CD19-A": np.random.lognormal(3, 0.8),
                            "CD56-A": np.random.lognormal(3.5, 0.8),
                        }

                        # Add some realistic correlations
                        event["FSC-H"] *= (1 + np.random.normal(0, 0.05))

                        data.append(event)

                    # Write data
                    for event in data:
                        line = ",".join(f"{v".3f"}" for v in event.values()) + "\n"
                        f.write(line)

                    # Wait for next batch
                    time.sleep(1.0)

        # Start simulation in thread
        import threading
        self._simulation_thread = threading.Thread(target=simulate, daemon=True)
        self._simulation_thread.start()

        print(f"🎭 Started data simulation: {self.events_per_second} events/sec for {duration_seconds}s")

    def stop_simulation(self) -> None:
        """Stop data simulation."""
        self._running = False
        if hasattr(self, '_simulation_thread'):
            self._simulation_thread.join(timeout=5.0)
        print("🛑 Data simulation stopped")


class LiveQualityMonitor:
    """Real-time quality monitoring for streaming data."""

    def __init__(self, alert_thresholds: dict[str, float] | None = None):
        """Initialize live quality monitor.

        Args:
            alert_thresholds: Dictionary of metric names to threshold values
        """
        self.alert_thresholds = alert_thresholds or {
            "debris_fraction": 0.3,
            "doublet_fraction": 0.2,
            "cv_fsc": 0.15,
            "cv_ssc": 0.15,
        }
        self._recent_metrics = deque(maxlen=100)
        self._alerts = []

    def process_qc_metrics(self, metrics: dict[str, float]) -> list[dict[str, Any]]:
        """Process QC metrics and generate alerts.

        Args:
            metrics: Dictionary of QC metrics

        Returns:
            List of generated alerts
        """
        alerts = []
        self._recent_metrics.append(metrics)

        # Check each metric against thresholds
        for metric, threshold in self.alert_thresholds.items():
            if metric in metrics:
                value = metrics[metric]

                if value > threshold:
                    alert = {
                        "type": "qc_threshold_exceeded",
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "severity": "warning" if value < threshold * 1.5 else "error",
                        "timestamp": pd.Timestamp.now().isoformat()
                    }

                    self._alerts.append(alert)
                    alerts.append(alert)

        # Keep only recent alerts
        if len(self._alerts) > 50:
            self._alerts = self._alerts[-50:]

        return alerts

    def get_recent_metrics(self) -> list[dict[str, float]]:
        """Get recent QC metrics."""
        return list(self._recent_metrics)

    def get_alerts(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent alerts."""
        return self._alerts[-limit:]


def create_realtime_pipeline(
    ws_url: str,
    output_dir: str | Path,
    qc_config: dict[str, Any] | None = None
) -> tuple[WebSocketProcessor, StreamingProcessor, LiveQualityMonitor]:
    """Create a complete real-time processing pipeline.

    Args:
        ws_url: WebSocket URL for data source
        output_dir: Directory for output files
        qc_config: Quality control configuration

    Returns:
        Tuple of (websocket_processor, streaming_processor, quality_monitor)
    """
    # Initialize components
    ws_processor = WebSocketProcessor(ws_url)
    streaming_processor = StreamingProcessor(output_dir)
    quality_monitor = LiveQualityMonitor()

    # Add QC monitoring to streaming processor
    def qc_handler(data: pd.DataFrame) -> dict[str, Any]:
        """QC analysis handler."""
        from .qc import add_qc_flags, qc_summary

        try:
            # Apply QC
            qc_data = add_qc_flags(data, qc_config)

            # Calculate metrics
            samples = {"live": qc_data}
            summary = qc_summary(samples)

            if not summary.empty:
                metrics = summary.iloc[0].to_dict()
                alerts = quality_monitor.process_qc_metrics(metrics)

                return {
                    "qc_metrics": metrics,
                    "alerts": alerts,
                    "data_quality": "good" if not alerts else "warning"
                }
        except Exception as e:
            print(f"Error in QC handler: {e}")

        return {"qc_metrics": {}, "alerts": []}

    streaming_processor.add_analysis_handler(qc_handler)

    # Add data quality monitoring
    def quality_monitor_handler(data: pd.DataFrame) -> dict[str, Any]:
        """Quality monitoring handler."""
        try:
            # Calculate basic quality metrics
            numeric_data = data.select_dtypes(include=[np.number])

            if not numeric_data.empty:
                cv_metrics = {}
                for col in numeric_data.columns:
                    if len(numeric_data[col]) > 1:
                        cv = numeric_data[col].std() / numeric_data[col].mean()
                        cv_metrics[f"cv_{col}"] = cv

                return {"cv_metrics": cv_metrics}

        except Exception as e:
            print(f"Error in quality monitor handler: {e}")

        return {"cv_metrics": {}}

    streaming_processor.add_analysis_handler(quality_monitor_handler)

    return ws_processor, streaming_processor, quality_monitor


def start_realtime_monitoring(
    ws_url: str,
    output_dir: str | Path,
    qc_config: dict[str, Any] | None = None,
    monitoring_port: int = 8081
) -> None:
    """Start complete real-time monitoring system.

    Args:
        ws_url: WebSocket URL for data source
        output_dir: Directory for output files
        qc_config: Quality control configuration
        monitoring_port: Port for monitoring dashboard
    """
    # Create pipeline components
    ws_processor, streaming_processor, quality_monitor = create_realtime_pipeline(
        ws_url, output_dir, qc_config
    )

    # Add monitoring dashboard updates
    def monitoring_update(data: pd.DataFrame) -> dict[str, Any]:
        """Update monitoring dashboard."""
        try:
            # Get current metrics
            rolling_stats = streaming_processor.get_rolling_stats()
            alerts = quality_monitor.get_alerts(5)

            # Update monitor
            monitor_data = {
                "data_rate": len(data),
                "buffer_size": len(ws_processor.buffer),
                "rolling_stats": rolling_stats,
                "recent_alerts": alerts,
                "processing_status": "active"
            }

            quality_monitor.update_monitoring_data(monitor_data)

            return {"monitoring": monitor_data}

        except Exception as e:
            print(f"Error in monitoring update: {e}")

        return {}

    streaming_processor.add_analysis_handler(monitoring_update)

    # Start processing
    ws_processor.start_processing()

    print("🚀 Real-time monitoring started")
    print(f"📊 Monitoring dashboard available at: http://localhost:{monitoring_port}")
    print(f"📁 Output directory: {output_dir}")

    # Keep running
    try:
        while ws_processor._running:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n🛑 Stopping real-time monitoring...")
        ws_processor.stop_processing()


def simulate_instrument_data(
    output_file: str | Path,
    duration_minutes: int = 5,
    events_per_second: int = 1000
) -> DataStreamSimulator:
    """Simulate flow cytometer data stream.

    Args:
        output_file: File to write simulated data
        duration_minutes: Duration of simulation
        events_per_second: Rate of data generation

    Returns:
        DataStreamSimulator instance
    """
    simulator = DataStreamSimulator(output_file, events_per_second)
    simulator.start_simulation(duration_minutes * 60)
    return simulator


def create_monitoring_dashboard(monitor: RealTimeMonitor, port: int = 8081) -> None:
    """Create a monitoring dashboard for real-time data.

    Args:
        monitor: RealTimeMonitor instance
        port: Port for dashboard
    """
    # This would create a Streamlit dashboard for monitoring
    # For now, it's a placeholder
    print(f"📊 Monitoring dashboard would be available at: http://localhost:{port}")


# Example usage functions
def demo_realtime_processing() -> None:
    """Demonstrate real-time processing capabilities."""
    print("🔬 CytoFlow-QC Real-Time Processing Demo")
    print("=" * 50)

    # Simulate data source
    simulator = simulate_instrument_data("/tmp/simulated_data.csv", duration_minutes=1)

    # Create real-time pipeline
    ws_processor, streaming_processor, quality_monitor = create_realtime_pipeline(
        "ws://localhost:8080", "/tmp/realtime_output"
    )

    # Start processing
    ws_processor.start_processing()

    print("🚀 Real-time processing demo started")
    print("📊 Processing simulated flow cytometry data...")

    # Let it run for a bit
    import time
    time.sleep(10)

    # Stop everything
    ws_processor.stop_processing()
    simulator.stop_simulation()

    print("✅ Demo completed")
    print(f"📁 Check output in: /tmp/realtime_output")


if __name__ == "__main__":
    demo_realtime_processing()

