
"""
Intel IoT MQTT Producer

Supports:
- Raw transmission
- Uniform Decimation
- Standard Deviation
- HP-PE
- HP-Energy
- Hybrid HP-PE + HP-Energy
- PID-based adaptive Hybrid downsampling

PID uses consumer feedback sent on MQTT topic q2.

Fixed mode:
    --algorithm hybrid --keep 0.2

Adaptive mode:
    --algorithm hybrid --keep 0.2 --pid
"""

import argparse
import csv
import json
import sys
import time
import uuid

from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock

import numpy as np
import paho.mqtt.client as mqtt

from config import get_config


# ============================================================
# 1. PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_all import load_intel
from algorithms import METHODS, choose_samples
from algorithms.pid_controller import PIDRetentionController


# ============================================================
# 2. MQTT CONFIGURATION
# ============================================================

config = get_config("mqtt", "edge")

BROKER = config.PARAMS["host"]
PORT = int(config.PARAMS["port"])

TOPIC = "q1"
FEEDBACK_TOPIC = "q2"

QOS = 1


# ============================================================
# 3. FAIR REPLAY TIMING
# ============================================================

def wait_until(target):

    remaining = target - time.perf_counter()

    if remaining > 0:
        time.sleep(remaining)


# ============================================================
# 4. MQTT CONNECTION
# ============================================================

def connect_mqtt(feedback_callback=None):

    connected = Event()
    subscribed = Event()

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

        if feedback_callback is not None:

            result, _ = client.subscribe(
                FEEDBACK_TOPIC,
                qos=1
            )

            if result != mqtt.MQTT_ERR_SUCCESS:
                print(
                    "Could not subscribe to feedback topic."
                )
                return

        connected.set()

    def on_subscribe(
        client,
        userdata,
        mid,
        reason_codes,
        properties
    ):

        subscribed.set()

    client.on_connect = on_connect
    client.on_subscribe = on_subscribe

    if feedback_callback is not None:
        client.on_message = feedback_callback

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
            "Could not connect to MQTT broker."
        )

    if feedback_callback is not None:

        if not subscribed.wait(timeout=10):

            client.disconnect()
            client.loop_stop()

            raise ConnectionError(
                "Could not subscribe to q2 feedback."
            )

    return client


# ============================================================
# 5. MQTT PUBLISHING
# ============================================================

def publish_message(client, payload):

    body = json.dumps(
        payload,
        separators=(",", ":")
    ).encode("utf-8")

    info = client.publish(
        TOPIC,
        payload=body,
        qos=QOS,
        retain=False
    )

    if info.rc != mqtt.MQTT_ERR_SUCCESS:

        raise RuntimeError(
            f"MQTT publish failed: {info.rc}"
        )

    # Wait for MQTT QoS 1 broker acknowledgment.
    info.wait_for_publish(timeout=10)

    if not info.is_published():

        raise TimeoutError(
            "MQTT QoS 1 publish acknowledgement timed out."
        )

    return len(body)


# ============================================================
# 6. SELECT SENSOR READINGS
# ============================================================

def choose_indices(values, args, retention):

    """
    Select readings from a buffered sensor block.

    Fixed mode:
        Process all available readings.

    PID mode:
        Process a block using the current retention budget.
    """

    n = len(values)

    if args.algorithm == "raw" or n < 8:

        return np.arange(
            n,
            dtype=int
        )

    # Prevent the filter's window sizes from exceeding
    # the current buffered block.
    local_window = min(
        args.window,
        n
    )

    local_chunk = min(
        args.chunk,
        n
    )

    selected = np.asarray(
        choose_samples(
            values,
            args.algorithm,
            retention,
            local_window,
            local_chunk,
            args.hp_alpha,
            args.pe_weight
        ),
        dtype=int
    )

    expected = min(
        n,
        max(
            2,
            round(n * retention)
        )
    )

    if (
        selected.ndim != 1
        or len(selected) != expected
        or np.any(selected < 0)
        or np.any(selected >= n)
        or len(np.unique(selected)) != len(selected)
    ):

        raise ValueError(
            "Downsampling algorithm returned invalid indices."
        )

    return np.sort(selected)


# ============================================================
# 7. SAVE EXPERIMENT RESULTS
# ============================================================

def save_publisher_results(
    run_id,
    summary,
    pid_history
):

    output_folder = (
        PROJECT_ROOT
        / "mqtt_results"
        / run_id
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_path = (
        output_folder / "publisher_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )

    if pid_history:

        history_path = (
            output_folder / "pid_history.csv"
        )

        with history_path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(pid_history[0])
            )

            writer.writeheader()
            writer.writerows(pid_history)

    print(
        "Publisher results saved to:",
        output_folder.resolve()
    )


# ============================================================
# 8. START PUBLISHING
# ============================================================

def start_publishing(args):

    # --------------------------------------------------------
    # LOAD INTEL SENSOR DATA
    # --------------------------------------------------------

    data_path = Path(args.data)

    if not data_path.is_absolute():

        data_path = (
            PROJECT_ROOT / data_path
        )

    print("\nLoading Intel Lab dataset...")

    df = load_intel(
        data_path,
        args.sensor,
        args.feature,
        args.max_samples
    )

    values = df["value"].to_numpy(
        dtype=float
    )

    original_count = len(values)

    if original_count == 0:

        raise ValueError(
            "No sensor readings available."
        )

    # --------------------------------------------------------
    # INITIALIZE PID OR FIXED DOWNSAMPLING
    # --------------------------------------------------------

    if args.pid:

        controller = PIDRetentionController(
            target_delay_ms=args.pid_target_delay_ms,
            initial_keep=args.keep,
            min_keep=args.pid_min_keep,
            max_keep=args.pid_max_keep,
            kp=args.pid_kp,
            ki=args.pid_ki,
            kd=args.pid_kd,
            smoothing=args.pid_smoothing
        )

        fixed_indices = None

    else:

        controller = None

        fixed_indices = choose_indices(
            values,
            args,
            args.keep
        )

    # --------------------------------------------------------
    # EXPERIMENT IDENTIFIER AND FEEDBACK STATE
    # --------------------------------------------------------

    run_id = uuid.uuid4().hex[:10]

    feedback_queue = Queue()

    sent_stamps = {}
    seen_feedback = set()

    feedback_lock = Lock()

    # --------------------------------------------------------
    # RECEIVE CONSUMER FEEDBACK FROM q2
    # --------------------------------------------------------

    def on_feedback(
        client,
        userdata,
        message
    ):

        if message.topic != FEEDBACK_TOPIC:
            return

        try:

            ack = json.loads(
                message.payload.decode("utf-8")
            )

            if (
                ack.get("run_id") != run_id
                or ack.get("type") != "ack"
            ):
                return

            message_id = int(
                ack["id"]
            )

            consumer_received_ns = int(
                ack["received_ns"]
            )

            with feedback_lock:

                if message_id in seen_feedback:
                    return

                sent_ns = sent_stamps.get(
                    message_id
                )

                if sent_ns is None:
                    return

                seen_feedback.add(
                    message_id
                )

            # Both applications currently run on the
            # same Windows computer.
            delay_ms = (
                consumer_received_ns - sent_ns
            ) / 1_000_000

            if 0 <= delay_ms < 120_000:

                feedback_queue.put(
                    (message_id, delay_ms)
                )

        except (
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError
        ):
            return

    # --------------------------------------------------------
    # EXPERIMENT INFORMATION
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print("INTEL IoT MQTT EXPERIMENT")
    print("=" * 60)

    print("Run ID:", run_id)
    print("Sensor:", args.sensor)
    print("Feature:", args.feature)
    print("Algorithm:", args.algorithm)

    print(
        "Mode:",
        "PID adaptive" if args.pid else "Fixed retention"
    )

    print(
        "Initial/Fixed retention:",
        f"{args.keep * 100:.1f}%"
    )

    print(
        "Original readings:",
        original_count
    )

    if not args.pid:

        print(
            "Selected readings:",
            len(fixed_indices)
        )

    print(
        "Replay interval:",
        f"{args.interval:.3f} s"
    )

    print(
        "MQTT endpoint:",
        f"{BROKER}:{PORT}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # CONNECT TO MQTT
    # --------------------------------------------------------

    client = connect_mqtt(
        on_feedback if args.pid else None
    )

    messages_sent = 0
    total_payload_bytes = 0

    scheduling_delays = []
    pid_history = []

    replay_start = time.perf_counter()

    # ========================================================
    # SEND SELECTED READINGS
    # ========================================================

    def send_selected(indices):

        nonlocal messages_sent
        nonlocal total_payload_bytes

        message_ids = []

        for index in indices:

            index = int(index)

            target_time = (
                replay_start
                + index * args.interval
            )

            wait_until(
                target_time
            )

            actual_start = time.perf_counter()

            scheduling_delay_ms = max(
                0.0,
                (
                    actual_start - target_time
                ) * 1000
            )

            scheduling_delays.append(
                scheduling_delay_ms
            )

            row = df.iloc[index]

            message_id = messages_sent + 1

            sent_ns = time.time_ns()

            payload = {
                "id": message_id,
                "data": float(row["value"]),
                "run_id": run_id,
                "algorithm": args.algorithm,
                "sensor_id": args.sensor,
                "feature": args.feature,
                "sample_index": index,
                "sensor_timestamp": str(
                    row["timestamp"]
                ),
                "sent_ns": sent_ns
            }

            # Associate the message with its timestamp
            # before the q2 acknowledgment can arrive.
            if args.pid:

                with feedback_lock:

                    sent_stamps[message_id] = sent_ns

            payload_size = publish_message(
                client,
                payload
            )

            messages_sent += 1

            total_payload_bytes += payload_size

            message_ids.append(
                message_id
            )

            print(
                f"Sent {message_id:03d} | "
                f"Index {index:04d} | "
                f"Value {float(row['value']):.3f} | "
                f"Payload {payload_size} B"
            )

        return message_ids

    # ========================================================
    # EXECUTE MQTT EXPERIMENT
    # ========================================================

    try:

        # ----------------------------------------------------
        # FIXED-RETENTION MODE
        # ----------------------------------------------------

        if not args.pid:

            send_selected(
                fixed_indices
            )

        # ----------------------------------------------------
        # PID-ADAPTIVE MODE
        # ----------------------------------------------------

        else:

            for chunk_number, start in enumerate(
                range(
                    0,
                    original_count,
                    args.pid_chunk_samples
                ),
                start=1
            ):

                end = min(
                    original_count,
                    start + args.pid_chunk_samples
                )

                # Retention chosen from previous feedback.
                current_retention = controller.keep

                chunk_values = values[start:end]

                relative_indices = choose_indices(
                    chunk_values,
                    args,
                    current_retention
                )

                selected_indices = (
                    start + relative_indices
                )

                print(
                    f"\nPID chunk {chunk_number} | "
                    f"Readings [{start}, {end}) | "
                    f"Retention {current_retention * 100:.1f}% | "
                    f"Selected {len(selected_indices)}/{end-start}"
                )

                # Transmit the readings selected for
                # this particular chunk.
                message_ids = send_selected(
                    selected_indices
                )

                # Maintain the original replay horizon
                # for the entire buffered chunk.
                chunk_end_time = (
                    replay_start
                    + (end - 1) * args.interval
                )

                wait_until(
                    chunk_end_time
                )

                # --------------------------------------------
                # COLLECT q2 FEEDBACK
                # --------------------------------------------

                pending = set(
                    message_ids
                )

                measured_delays = []

                deadline = (
                    time.perf_counter()
                    + args.pid_feedback_wait
                )

                while pending:

                    remaining = (
                        deadline - time.perf_counter()
                    )

                    if remaining <= 0:
                        break

                    try:

                        ack_id, delay_ms = (
                            feedback_queue.get(
                                timeout=remaining
                            )
                        )

                    except Empty:
                        break

                    if ack_id in pending:

                        pending.remove(
                            ack_id
                        )

                        measured_delays.append(
                            delay_ms
                        )

                # --------------------------------------------
                # UPDATE PID RETENTION
                # --------------------------------------------

                avg_delay = (
                    float(np.mean(measured_delays))
                    if measured_delays
                    else None
                )

                if avg_delay is not None:

                    dt_seconds = (
                        end - start
                    ) * args.interval

                    next_retention = controller.update(
                        measured_delay_ms=avg_delay,
                        dt_seconds=dt_seconds
                    )

                    print(
                        f"PID feedback: "
                        f"{len(measured_delays)}/{len(message_ids)} ACKs | "
                        f"Mean delay {avg_delay:.3f} ms | "
                        f"Next retention {next_retention * 100:.1f}%"
                    )

                else:

                    print(
                        "WARNING: No q2 feedback received "
                        "for this chunk. Retention unchanged."
                    )

                # --------------------------------------------
                # LOG PID DECISION
                # --------------------------------------------

                pid_history.append({
                    "run_id": run_id,
                    "chunk_number": chunk_number,
                    "start_index": start,
                    "end_index_exclusive": end,
                    "original_samples": end - start,
                    "selected_samples": len(
                        selected_indices
                    ),
                    "requested_retention_pct": round(
                        current_retention * 100,
                        4
                    ),
                    "actual_retention_pct": round(
                        len(selected_indices)
                        / (end - start)
                        * 100,
                        4
                    ),
                    "feedback_acks": len(
                        measured_delays
                    ),
                    "expected_acks": len(
                        message_ids
                    ),
                    "feedback_delivery_delay_ms": avg_delay,
                    "next_retention_pct": round(
                        controller.keep * 100,
                        4
                    )
                })

        # ----------------------------------------------------
        # COMPLETE THE REPLAY HORIZON
        # ----------------------------------------------------

        replay_end_target = (
            replay_start
            + (original_count - 1) * args.interval
        )

        wait_until(
            replay_end_target
        )

        replay_duration = (
            time.perf_counter() - replay_start
        )

        # ----------------------------------------------------
        # SEND EXPERIMENT END MARKER
        # ----------------------------------------------------

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
    # CALCULATE FINAL RESULTS
    # ========================================================

    reduction_pct = (
        100 * (
            1 - messages_sent / original_count
        )
    )

    average_publish_rate = (
        messages_sent / replay_duration
    )

    application_payload_rate = (
        total_payload_bytes / replay_duration
    )

    average_scheduling_delay = float(
        np.mean(scheduling_delays)
    )

    maximum_scheduling_delay = float(
        np.max(scheduling_delays)
    )

    summary = {
        "run_id": run_id,
        "algorithm": args.algorithm,
        "mode": (
            "pid_adaptive"
            if args.pid
            else "fixed"
        ),
        "sensor_id": args.sensor,
        "feature": args.feature,
        "original_readings": original_count,
        "messages_published": messages_sent,
        "data_reduction_pct": reduction_pct,
        "telemetry_payload_bytes": total_payload_bytes,
        "replay_duration_seconds": replay_duration,
        "average_publish_rate_messages_per_sec": (
            average_publish_rate
        ),
        "application_payload_rate_bytes_per_sec": (
            application_payload_rate
        ),
        "average_scheduling_delay_ms": (
            average_scheduling_delay
        ),
        "maximum_scheduling_delay_ms": (
            maximum_scheduling_delay
        ),
        "initial_retention_pct": args.keep * 100,
        "pid_target_delay_ms": (
            args.pid_target_delay_ms
            if args.pid
            else None
        ),
        "pid_chunk_samples": (
            args.pid_chunk_samples
            if args.pid
            else None
        )
    }

    save_publisher_results(
        run_id,
        summary,
        pid_history
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 60)

    print("MQTT PUBLISHER SUMMARY")
    print("=" * 60)

    print("Run ID:", run_id)
    print("Algorithm:", args.algorithm)
    print("Mode:", summary["mode"])

    print(
        "Original readings:",
        original_count
    )

    print(
        "Messages published:",
        messages_sent
    )

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

    print(
        f"Average publish rate: "
        f"{average_publish_rate:.2f} messages/sec"
    )

    print(
        f"Application payload rate: "
        f"{application_payload_rate:.2f} bytes/sec"
    )

    print(
        f"Average scheduling delay: "
        f"{average_scheduling_delay:.3f} ms"
    )

    print(
        f"Maximum scheduling delay: "
        f"{maximum_scheduling_delay:.3f} ms"
    )

    print("=" * 60)


# ============================================================
# 9. COMMAND-LINE ARGUMENTS
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Intel Lab MQTT producer with "
            "fixed or PID-adaptive retention"
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
        default=0.2
    )

    # --------------------------------------------------------
    # PID OPTIONS
    # --------------------------------------------------------

    parser.add_argument(
        "--pid",
        action="store_true",
        help="Enable adaptive retention for Hybrid"
    )

    parser.add_argument(
        "--pid-chunk-samples",
        type=int,
        default=20
    )

    parser.add_argument(
        "--pid-feedback-wait",
        type=float,
        default=0.75
    )

    parser.add_argument(
        "--pid-target-delay-ms",
        type=float,
        default=100.0
    )

    parser.add_argument(
        "--pid-min-keep",
        type=float,
        default=0.10
    )

    parser.add_argument(
        "--pid-max-keep",
        type=float,
        default=0.50
    )

    parser.add_argument(
        "--pid-kp",
        type=float,
        default=0.0015
    )

    parser.add_argument(
        "--pid-ki",
        type=float,
        default=0.00003
    )

    parser.add_argument(
        "--pid-kd",
        type=float,
        default=0.00001
    )

    parser.add_argument(
        "--pid-smoothing",
        type=float,
        default=0.4
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

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
            "--keep must be in (0, 1]."
        )

    if args.window < 6 or args.chunk < 8:

        parser.error(
            "--window must be >= 6 and --chunk must be >= 8."
        )

    if (
        not 0 < args.hp_alpha <= 1
        or not 0 <= args.pe_weight <= 1
    ):

        parser.error(
            "Invalid high-pass algorithm parameters."
        )

    if args.interval <= 0:

        parser.error(
            "--interval must be positive."
        )

    if args.pid:

        if args.algorithm != "hybrid":

            parser.error(
                "For the first PID experiment, "
                "use --algorithm hybrid."
            )

        if args.pid_chunk_samples < 8:

            parser.error(
                "--pid-chunk-samples must be >= 8."
            )

        if args.pid_feedback_wait < 0:

            parser.error(
                "--pid-feedback-wait cannot be negative."
            )

        if not (
            0 < args.pid_min_keep
            <= args.keep
            <= args.pid_max_keep
            <= 1
        ):

            parser.error(
                "Require 0 < pid-min-keep <= keep "
                "<= pid-max-keep <= 1."
            )

    start_publishing(
        args
    )


# ============================================================
# 10. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()