
"""
Intel IoT MQTT Producer
Adapted from the Vicomtech producer.

Features:
- Loads real Intel Lab sensor readings
- Supports Raw and five downsampling algorithms
- Publishes selected samples through MQTT
- Uses fair replay timing based on original sample positions
- Measures application payload bytes
- Uses the existing producer config.py
- Compatible with Paho MQTT 2.x

MQTT Topic: q1
Broker: Local Mosquitto

NOTE:
Algorithms still select samples offline before replay.
This is not yet a live adaptive streaming implementation.
"""

import argparse
import json
import sys
import time
import uuid

from pathlib import Path
from threading import Event

import numpy as np
import paho.mqtt.client as mqtt

from config import get_config


# ============================================================
# 1. IMPORT PROJECT MODULES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from benchmark_all import load_intel
from algorithms import METHODS, choose_samples


# ============================================================
# 2. MQTT CONFIGURATION
# ============================================================

config = get_config("mqtt", "edge")

BROKER = config.PARAMS["host"]
PORT = int(config.PARAMS["port"])

TOPIC = "q1"
QOS = 1


# ============================================================
# 3. LOAD AND DOWNSAMPLE INTEL SENSOR DATA
# ============================================================

def prepare_data(args):

    dataset_path = Path(args.data)

    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    print("\nLoading Intel Lab dataset...")

    df = load_intel(
        dataset_path,
        args.sensor,
        args.feature,
        args.max_samples
    )

    values = df["value"].to_numpy(dtype=float)

    original_count = len(values)

    # Raw baseline: transmit every reading.
    if args.algorithm == "raw":

        indices = np.arange(
            original_count,
            dtype=int
        )

    # Apply a selected downsampling algorithm.
    else:

        indices = choose_samples(
            values,
            args.algorithm,
            args.keep,
            args.window,
            args.chunk,
            args.hp_alpha,
            args.pe_weight
        )

        indices = np.asarray(
            indices,
            dtype=int
        )

    # Validate indices before transmission.
    expected = (
        original_count
        if args.algorithm == "raw"
        else max(
            2,
            min(
                original_count,
                round(original_count * args.keep)
            )
        )
    )

    if (
        indices.ndim != 1
        or not np.issubdtype(
            indices.dtype,
            np.integer
        )
        or len(indices) != expected
        or np.any(indices < 0)
        or np.any(indices >= original_count)
        or len(np.unique(indices)) != len(indices)
    ):

        raise ValueError(
            "Algorithm returned invalid sample indices."
        )

    indices = np.sort(indices.astype(int))

    return df, indices


# ============================================================
# 4. CONNECT TO MOSQUITTO MQTT BROKER
# ============================================================

def connect_mqtt():

    connected = Event()

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="intel_edge_" + uuid.uuid4().hex[:8]
    )

    def on_connect(
        client,
        userdata,
        flags,
        reason_code,
        properties
    ):

        if reason_code.is_failure:

            print(
                "MQTT connection failed:",
                reason_code
            )

            return

        print("Connected to MQTT broker successfully.")

        connected.set()

    client.on_connect = on_connect

    print(
        f"Connecting to MQTT broker {BROKER}:{PORT}..."
    )

    client.connect(
        BROKER,
        PORT,
        keepalive=60
    )

    client.loop_start()

    if not connected.wait(timeout=10):

        client.disconnect()
        client.loop_stop()

        raise ConnectionError(
            "Could not establish MQTT connection."
        )

    return client


# ============================================================
# 5. PUBLISH MQTT MESSAGE
# ============================================================

def publish_message(client, payload):

    body = json.dumps(
        payload,
        separators=(",", ":")
    ).encode("utf-8")

    result = client.publish(
        TOPIC,
        payload=body,
        qos=QOS,
        retain=False
    )

    if result.rc != mqtt.MQTT_ERR_SUCCESS:

        raise RuntimeError(
            f"MQTT publish failed: {result.rc}"
        )

    # Wait for broker acknowledgement (QoS 1).
    result.wait_for_publish(timeout=10)

    if not result.is_published():

        raise TimeoutError(
            "MQTT publish acknowledgement timed out."
        )

    return len(body)


# ============================================================
# 6. REPLAY TIMING
# ============================================================

def wait_until(target_time):
    """
    Wait until a scheduled monotonic-clock timestamp.

    The target is based on the reading's ORIGINAL
    index, not its position among retained samples.
    """

    remaining = target_time - time.perf_counter()

    if remaining > 0:
        time.sleep(remaining)


# ============================================================
# 7. RUN MQTT PUBLISHING EXPERIMENT
# ============================================================

def start_publishing(args):

    # --------------------------------------------------------
    # PREPARE INPUT DATA
    # --------------------------------------------------------

    df, indices = prepare_data(args)

    original_count = len(df)
    retained_count = len(indices)

    reduction_pct = (
        100 * (
            1 - retained_count / original_count
        )
    )

    # Unique identifier for each MQTT experiment.
    run_id = uuid.uuid4().hex[:10]

    print("\n" + "=" * 55)
    print("INTEL IoT MQTT EXPERIMENT")
    print("=" * 55)

    print("Run ID:", run_id)
    print("Sensor:", args.sensor)
    print("Feature:", args.feature)
    print("Algorithm:", args.algorithm)

    if args.algorithm != "raw":
        print(
            f"Target retention: {args.keep * 100:.1f}%"
        )

    print("Original readings:", original_count)
    print("Selected readings:", retained_count)

    print(
        f"Data reduction: {reduction_pct:.2f}%"
    )

    print(
        f"Replay interval: {args.interval:.3f} seconds"
    )

    print(
        "Original replay horizon:",
        f"{(original_count - 1) * args.interval:.2f} s"
    )

    print("MQTT Broker:", BROKER)
    print("MQTT Topic:", TOPIC)

    print("=" * 55)

    # --------------------------------------------------------
    # CONNECT TO MQTT
    # --------------------------------------------------------

    client = connect_mqtt()

    total_payload_bytes = 0
    messages_sent = 0
    schedule_lags_ms = []

    # Start the replay clock AFTER preparing data
    # and connecting to MQTT.
    replay_start = time.perf_counter()

    try:

        # ====================================================
        # REPLAY SELECTED READINGS
        # ====================================================

        for sequence, index in enumerate(
            indices,
            start=1
        ):

            index = int(index)

            # ------------------------------------------------
            # FAIR REPLAY SCHEDULING
            # ------------------------------------------------

            # Every method uses the original sample index.
            #
            # Example with interval = 0.2:
            #
            # Original index 0   -> 0.0 seconds
            # Original index 10  -> 2.0 seconds
            # Original index 50  -> 10.0 seconds
            # Original index 99  -> 19.8 seconds
            #
            # Raw and Hybrid share the same time horizon.

            target_time = (
                replay_start
                + index * args.interval
            )

            wait_until(target_time)

            actual_start = time.perf_counter()

            lag_ms = max(
                0.0,
                (actual_start - target_time) * 1000
            )

            schedule_lags_ms.append(lag_ms)

            # ------------------------------------------------
            # READ ORIGINAL SENSOR VALUE
            # ------------------------------------------------

            row = df.iloc[index]

            # ------------------------------------------------
            # BUILD MQTT PAYLOAD
            # ------------------------------------------------

            payload = {
                "id": sequence,
                "data": float(row["value"]),
                "run_id": run_id,
                "algorithm": args.algorithm,
                "sensor_id": args.sensor,
                "feature": args.feature,
                "sample_index": index,
                "sensor_timestamp": str(
                    row["timestamp"]
                ),
                "sent_ns": time.time_ns()
            }

            # ------------------------------------------------
            # PUBLISH READING THROUGH MQTT
            # ------------------------------------------------

            payload_bytes = publish_message(
                client,
                payload
            )

            total_payload_bytes += payload_bytes

            messages_sent += 1

            print(
                f"Sent {sequence:03d}/{retained_count:03d} | "
                f"Index {index:04d} | "
                f"Value {float(row['value']):.3f} | "
                f"Payload {payload_bytes} B"
            )

        # ====================================================
        # COMPLETE THE ORIGINAL REPLAY HORIZON
        # ====================================================

        # This also handles methods that do not retain
        # the last reading.
        replay_end_target = (
            replay_start
            + (original_count - 1) * args.interval
        )

        wait_until(replay_end_target)

        replay_duration = (
            time.perf_counter() - replay_start
        )

        # ====================================================
        # SEND EXPERIMENT END MARKER
        # ====================================================

        # The existing Vicomtech consumer recognises id=-1
        # and writes its results to disk.

        end_payload = {
            "id": -1,
            "data": 1,
            "type": "experiment_end",
            "run_id": run_id,
            "algorithm": args.algorithm,
            "messages_sent": messages_sent,
            "sent_ns": time.time_ns()
        }

        publish_message(
            client,
            end_payload
        )

    finally:

        client.disconnect()
        client.loop_stop()

    # ========================================================
    # 8. DISPLAY PUBLISHER SUMMARY
    # ========================================================

    print("\n" + "=" * 55)

    print("MQTT PUBLISHER SUMMARY")

    print("=" * 55)

    print("Run ID:", run_id)
    print("Algorithm:", args.algorithm)

    print("Original readings:", original_count)
    print("Messages published:", messages_sent)

    print(
        f"Data reduction: {reduction_pct:.2f}%"
    )

    print(
        "Telemetry payload bytes:",
        total_payload_bytes
    )

    print(
        f"Replay duration: {replay_duration:.3f} seconds"
    )

    if replay_duration > 0:

        effective_rate = (
            messages_sent / replay_duration
        )

        payload_rate = (
            total_payload_bytes / replay_duration
        )

        print(
            "Average publish rate:",
            f"{effective_rate:.2f} messages/sec"
        )

        print(
            "Application payload rate:",
            f"{payload_rate:.2f} bytes/sec"
        )

    if schedule_lags_ms:

        print(
            "Average scheduling delay:",
            f"{np.mean(schedule_lags_ms):.3f} ms"
        )

        print(
            "Maximum scheduling delay:",
            f"{np.max(schedule_lags_ms):.3f} ms"
        )

    print("=" * 55)

    print("\nMQTT publishing completed.")


# ============================================================
# 9. COMMAND-LINE ARGUMENTS
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Intel Lab MQTT producer with fair replay timing"
        )
    )

    parser.add_argument(
        "--sensor",
        type=int,
        default=1
    )

    parser.add_argument(
        "--feature",
        choices=[
            "temperature",
            "humidity",
            "light",
            "voltage"
        ],
        default="temperature"
    )

    parser.add_argument(
        "--data",
        default="data/data.txt.gz"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=100
    )

    parser.add_argument(
        "--algorithm",
        choices=["raw"] + list(METHODS),
        default="raw"
    )

    parser.add_argument(
        "--keep",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--window",
        type=int,
        default=48
    )

    parser.add_argument(
        "--chunk",
        type=int,
        default=256
    )

    parser.add_argument(
        "--hp-alpha",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--pe-weight",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help=(
            "Seconds between original sensor sample positions. "
            "All algorithms use the same replay horizon."
        )
    )

    args = parser.parse_args()

    # ========================================================
    # VALIDATE INPUT SETTINGS
    # ========================================================

    if not 1 <= args.sensor <= 54:

        parser.error(
            "--sensor must be between 1 and 54."
        )

    if args.max_samples < 100:

        parser.error(
            "--max-samples must be at least 100."
        )

    if not 0 < args.keep <= 1:

        parser.error(
            "--keep must be between 0 and 1."
        )

    if args.window < 6:

        parser.error(
            "--window must be at least 6."
        )

    if args.chunk < 8:

        parser.error(
            "--chunk must be at least 8."
        )

    if not 0 < args.hp_alpha <= 1:

        parser.error(
            "--hp-alpha must be in (0, 1]."
        )

    if not 0 <= args.pe_weight <= 1:

        parser.error(
            "--pe-weight must be in [0, 1]."
        )

    if args.interval <= 0:

        parser.error(
            "--interval must be greater than zero."
        )

    start_publishing(args)


# ============================================================
# 10. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()