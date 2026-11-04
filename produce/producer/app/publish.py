
"""Intel IoT MQTT producer: fixed, PID-adaptive, and synchronized recovery.

Normal fixed experiment:
  python produce/producer/app/publish.py --algorithm hybrid --keep 0.2

Matched synchronized recovery (300 readings):
  python produce/producer/app/publish.py --algorithm hybrid --keep 0.2 --max-samples 300 --interval 0.2 --blockwise-fixed --synced-recovery
  python produce/producer/app/publish.py --algorithm hybrid --keep 0.2 --max-samples 300 --interval 0.2 --pid --synced-recovery

For --synced-recovery, Toxiproxy automatically enables 1 KB/s upstream bandwidth
before replay and removes it at original sample 150 (30.0 s with interval=0.2).
"""

import argparse
import csv
import json
import sys
import time
import uuid
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import paho.mqtt.client as mqtt

from config import get_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from benchmark_all import load_intel
from algorithms import METHODS, choose_samples
from algorithms.pid_controller import PIDRetentionController

config = get_config("mqtt", "edge")
BROKER = config.PARAMS["host"]
PORT = int(config.PARAMS["port"])
TOPIC = "q1"
FEEDBACK_TOPIC = "q2"
QOS = 1

# The management API must remain bound to localhost.
TOXIPROXY_API = "http://127.0.0.1:8474"
PROXY_NAME = "mqtt_proxy"
BANDWIDTH_TOXIC = "bandwidth_up_1kb"


# ============================================================
# 1. TOXIPROXY API
# ============================================================

def proxy_api(method, path, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        TOXIPROXY_API + path,
        data=data,
        method=method,
        headers={
            "User-Agent": "IntelMQTTSync/1.0",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=5) as response:
            raw = response.read()
            return json.loads(raw) if raw else None
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Toxiproxy {method} {path} failed: {exc}") from exc


# ============================================================
# 2. ENABLE BANDWIDTH DEGRADATION
# ============================================================

def enable_synced_bandwidth():
    """Require exactly the intended proxy/toxic configuration."""
    if BROKER != "127.0.0.1" or PORT != 1884:
        raise RuntimeError(
            "Synchronized recovery requires producer MQTT host=127.0.0.1 "
            "and port=1884. Leave consumer on 1883."
        )

    details = proxy_api("GET", f"/proxies/{PROXY_NAME}")
    if not details or not details.get("enabled", True):
        raise RuntimeError("mqtt_proxy is missing or disabled.")
    if (details.get("listen") != "127.0.0.1:1884" or
            details.get("upstream") != "127.0.0.1:1883"):
        raise RuntimeError("mqtt_proxy must forward 127.0.0.1:1884 to 127.0.0.1:1883.")

    toxics = proxy_api("GET", f"/proxies/{PROXY_NAME}/toxics") or []
    unexpected = [
        toxic["name"] for toxic in toxics
        if toxic.get("name") != BANDWIDTH_TOXIC and toxic.get("toxicity", 1) > 0
    ]
    if unexpected:
        raise RuntimeError(f"Remove unrelated active Toxiproxy toxics first: {unexpected}")

    existing = next((t for t in toxics if t.get("name") == BANDWIDTH_TOXIC), None)
    if existing is not None:
        if (existing.get("type") != "bandwidth" or
                existing.get("stream") != "upstream" or
                int(existing.get("attributes", {}).get("rate", -1)) != 1 or
                float(existing.get("toxicity", 0)) != 1.0):
            raise RuntimeError(
                "Existing bandwidth_up_1kb has unexpected settings; "
                "inspect it before continuing."
            )
        print("Network profile: existing 1 KB/s upstream toxic verified.")
    else:
        proxy_api("POST", f"/proxies/{PROXY_NAME}/toxics", {
            "name": BANDWIDTH_TOXIC,
            "type": "bandwidth",
            "stream": "upstream",
            "toxicity": 1.0,
            "attributes": {"rate": 1},
        })
        print("Network profile: 1 KB/s upstream toxic enabled.")


# ============================================================
# 3. SYNCHRONIZED NETWORK RECOVERY
# ============================================================

def start_recovery_worker(replay_start, sample_number, interval, result, stop_event):
    """Transition on the monotonic replay clock, independent of MQTT sends."""
    target_elapsed = sample_number * interval
    target = replay_start + target_elapsed

    def worker():
        while True:
            remaining = target - time.perf_counter()
            if remaining <= 0:
                break
            if stop_event.wait(timeout=min(remaining, 0.25)):
                return

        result["delete_requested_elapsed_s"] = time.perf_counter() - replay_start
        try:
            proxy_api("DELETE", f"/proxies/{PROXY_NAME}/toxics/{BANDWIDTH_TOXIC}")
            result["delete_confirmed_elapsed_s"] = time.perf_counter() - replay_start
            result["success"] = True
            print(
                "\nNETWORK RECOVERED: bandwidth toxic removed at "
                f"{result['delete_confirmed_elapsed_s']:.3f}s "
                f"(target {target_elapsed:.3f}s)."
            )
        except Exception as exc:
            result["success"] = False
            result["error"] = str(exc)
            print("NETWORK RECOVERY FAILED:", exc)

    thread = Thread(target=worker, name="network-recovery", daemon=True)
    thread.start()
    return thread


# ============================================================
# 4. REPLAY TIMING
# ============================================================

def wait_until(target):
    remaining = target - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


# ============================================================
# 5. MQTT CONNECTION
# ============================================================

def connect_mqtt(feedback_callback=None):
    connected = Event()
    subscribed = Event()
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="intel_edge_" + uuid.uuid4().hex[:8],
    )

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            print("MQTT connection failed:", reason_code)
            return
        print("Connected to MQTT broker successfully.")
        if feedback_callback is not None:
            result, _ = client.subscribe(FEEDBACK_TOPIC, qos=1)
            if result != mqtt.MQTT_ERR_SUCCESS:
                print("Could not subscribe to q2 feedback.")
                return
        connected.set()

    def on_subscribe(client, userdata, mid, reason_codes, properties):
        subscribed.set()

    client.on_connect = on_connect
    client.on_subscribe = on_subscribe
    if feedback_callback is not None:
        client.on_message = feedback_callback

    print(f"Connecting to MQTT broker {BROKER}:{PORT}...")
    client.connect(BROKER, PORT, keepalive=60)
    client.loop_start()
    if not connected.wait(timeout=10) or (
        feedback_callback is not None and not subscribed.wait(timeout=10)
    ):
        client.disconnect()
        client.loop_stop()
        raise ConnectionError("MQTT connection or q2 subscription failed.")
    return client


# ============================================================
# 6. PUBLISH MQTT MESSAGE
# ============================================================

def publish_message(client, payload):
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    info = client.publish(TOPIC, payload=body, qos=QOS, retain=False)
    if info.rc != mqtt.MQTT_ERR_SUCCESS:
        raise RuntimeError(f"MQTT publish failed: {info.rc}")
    info.wait_for_publish(timeout=10)
    if not info.is_published():
        raise TimeoutError("MQTT broker acknowledgement timed out.")
    return len(body)


# ============================================================
# 7. SELECT SENSOR READINGS
# ============================================================

def choose_indices(values, args, retention):
    n = len(values)
    if args.algorithm == "raw":
        return np.arange(n, dtype=int)
    if n < 8:
        expected = min(n, max(2, round(n * retention)))
        return np.linspace(0, n - 1, expected).round().astype(int)

    result = np.asarray(
        choose_samples(
            values, args.algorithm, retention,
            min(args.window, n), min(args.chunk, n),
            args.hp_alpha, args.pe_weight,
        ),
        dtype=int,
    )
    expected = min(n, max(2, round(n * retention)))
    if (result.ndim != 1 or len(result) != expected or
            np.any(result < 0) or np.any(result >= n) or
            len(np.unique(result)) != len(result)):
        raise ValueError("Algorithm returned invalid sample indices.")
    return np.sort(result)


# ============================================================
# 8. SAVE RESULTS
# ============================================================

def save_results(run_id, summary, pid_history, network_event):
    folder = PROJECT_ROOT / "mqtt_results" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "publisher_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if pid_history:
        with (folder / "pid_history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(pid_history[0]))
            writer.writeheader()
            writer.writerows(pid_history)
    if network_event is not None:
        with (folder / "network_events.json").open("w", encoding="utf-8") as f:
            json.dump(network_event, f, indent=2)
    print("Publisher results saved to:", folder.resolve())


# ============================================================
# 9. MAIN MQTT EXPERIMENT
# ============================================================

def start_publishing(args):
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    print("\nLoading Intel Lab dataset...")
    df = load_intel(data_path, args.sensor, args.feature, args.max_samples)
    values = df["value"].to_numpy(dtype=float)
    count = len(values)
    if count == 0:
        raise ValueError("No sensor readings were loaded.")
    if args.synced_recovery and count <= args.recovery_sample:
        raise ValueError(
            f"Dataset has {count} readings; recovery sample {args.recovery_sample} "
            "must be within the replay."
        )

    controller = None
    if args.pid:
        controller = PIDRetentionController(
            target_delay_ms=args.pid_target_delay_ms,
            initial_keep=args.keep,
            min_keep=args.pid_min_keep,
            max_keep=args.pid_max_keep,
            kp=args.pid_kp,
            ki=args.pid_ki,
            kd=args.pid_kd,
            smoothing=args.pid_smoothing,
        )

    if not args.pid and not args.blockwise_fixed:
        fixed_indices = choose_indices(values, args, args.keep)
    else:
        fixed_indices = None

    run_id = uuid.uuid4().hex[:10]
    feedback_queue = Queue()
    sent_stamps = {}
    seen_feedback = set()
    feedback_lock = Lock()

    # --------------------------------------------------------
    # RECEIVE CONSUMER FEEDBACK
    # --------------------------------------------------------

    def on_feedback(client, userdata, message):
        if message.topic != FEEDBACK_TOPIC:
            return
        try:
            ack = json.loads(message.payload.decode("utf-8"))
            if ack.get("run_id") != run_id or ack.get("type") != "ack":
                return
            msg_id = int(ack["id"])
            consumer_ns = int(ack["received_ns"])
            with feedback_lock:
                if msg_id in seen_feedback:
                    return
                sent_ns = sent_stamps.get(msg_id)
                if sent_ns is None:
                    return
                seen_feedback.add(msg_id)
            delay_ms = (consumer_ns - sent_ns) / 1_000_000
            if 0 <= delay_ms < 120_000:
                feedback_queue.put((msg_id, delay_ms))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            return

    # --------------------------------------------------------
    # DISPLAY EXPERIMENT SETTINGS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("INTEL IoT MQTT EXPERIMENT")
    print("=" * 60)
    print("Run ID:", run_id)
    print("Algorithm:", args.algorithm)
    print("Mode:", "pid_adaptive" if args.pid else "fixed")
    print("Blockwise fixed:", args.blockwise_fixed)
    print("Sensor:", args.sensor, "Feature:", args.feature)
    print("Original readings:", count)
    print("Initial/Fixed retention:", f"{args.keep * 100:.1f}%")
    print("Replay interval:", f"{args.interval:.3f}s")
    print("MQTT endpoint:", f"{BROKER}:{PORT}")
    if args.synced_recovery:
        print(
            "Synchronized recovery: 1 KB/s -> unrestricted at "
            f"sample {args.recovery_sample} "
            f"({args.recovery_sample * args.interval:.3f}s after replay starts)"
        )
    print("=" * 60)

    # Network configuration is prepared BEFORE connecting and starting the replay.
    if args.synced_recovery:
        enable_synced_bandwidth()

    client = connect_mqtt(on_feedback if args.pid else None)
    messages_sent = 0
    payload_bytes = 0
    lags_ms = []
    pid_history = []
    network_event = None
    recovery_thread = None
    stop_event = Event()
    replay_start = time.perf_counter()

    if args.synced_recovery:
        network_event = {
            "run_id": run_id,
            "profile": "1 KB/s upstream then unrestricted",
            "recovery_sample": args.recovery_sample,
            "target_transition_elapsed_s": args.recovery_sample * args.interval,
            "delete_requested_elapsed_s": None,
            "delete_confirmed_elapsed_s": None,
            "success": False,
        }
        recovery_thread = start_recovery_worker(
            replay_start, args.recovery_sample, args.interval,
            network_event, stop_event,
        )

    # --------------------------------------------------------
    # SEND SELECTED SENSOR READINGS
    # --------------------------------------------------------

    def send_indices(indices):
        nonlocal messages_sent, payload_bytes
        sent_ids = []
        for index in indices:
            index = int(index)
            target = replay_start + index * args.interval
            wait_until(target)
            start = time.perf_counter()
            lags_ms.append(max(0.0, (start - target) * 1000))
            row = df.iloc[index]
            msg_id = messages_sent + 1
            sent_ns = time.time_ns()
            payload = {
                "id": msg_id,
                "data": float(row["value"]),
                "run_id": run_id,
                "algorithm": args.algorithm,
                "sensor_id": args.sensor,
                "feature": args.feature,
                "sample_index": index,
                "sensor_timestamp": str(row["timestamp"]),
                "sent_ns": sent_ns,
            }
            if args.pid:
                with feedback_lock:
                    sent_stamps[msg_id] = sent_ns
            size = publish_message(client, payload)
            messages_sent += 1
            payload_bytes += size
            sent_ids.append(msg_id)
            print(
                f"Sent {msg_id:03d} | Index {index:04d} | "
                f"Value {float(row['value']):.3f} | Payload {size} B"
            )
        return sent_ids

    # --------------------------------------------------------
    # EXECUTE MQTT EXPERIMENT
    # --------------------------------------------------------

    try:
        if not args.pid and not args.blockwise_fixed:
            send_indices(fixed_indices)
        else:
            for block_no, begin in enumerate(
                range(0, count, args.pid_chunk_samples), start=1
            ):
                end = min(count, begin + args.pid_chunk_samples)
                keep = controller.keep if args.pid else args.keep
                relative = choose_indices(values[begin:end], args, keep)
                selected = begin + relative
                print(
                    f"\nBlock {block_no} [{begin}, {end}) | "
                    f"Retention {keep * 100:.2f}% | "
                    f"Selected {len(selected)}/{end - begin}"
                )
                sent_ids = send_indices(selected)
                wait_until(replay_start + (end - 1) * args.interval)
                if not args.pid:
                    continue

                pending = set(sent_ids)
                delays = []
                deadline = time.perf_counter() + args.pid_feedback_wait
                while pending:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        ack_id, delay_ms = feedback_queue.get(timeout=remaining)
                    except Empty:
                        break
                    if ack_id in pending:
                        pending.remove(ack_id)
                        delays.append(delay_ms)

                avg_delay = float(np.mean(delays)) if delays else None
                if avg_delay is not None:
                    next_keep = controller.update(
                        measured_delay_ms=avg_delay,
                        dt_seconds=(end - begin) * args.interval,
                    )
                    print(
                        f"PID feedback: {len(delays)}/{len(sent_ids)} ACKs | "
                        f"Mean delay {avg_delay:.3f} ms | "
                        f"Next retention {next_keep * 100:.2f}%"
                    )
                else:
                    print("WARNING: No matching q2 ACKs; retention unchanged.")

                pid_history.append({
                    "run_id": run_id,
                    "chunk_number": block_no,
                    "start_index": begin,
                    "end_index_exclusive": end,
                    "original_samples": end - begin,
                    "selected_samples": len(selected),
                    "requested_retention_pct": round(keep * 100, 4),
                    "actual_retention_pct": round(len(selected) / (end - begin) * 100, 4),
                    "feedback_acks": len(delays),
                    "expected_acks": len(sent_ids),
                    "feedback_delivery_delay_ms": avg_delay,
                    "next_retention_pct": round(controller.keep * 100, 4),
                })

        wait_until(replay_start + (count - 1) * args.interval)
        replay_duration = time.perf_counter() - replay_start
        publish_message(client, {
            "id": -1,
            "data": 1,
            "type": "experiment_end",
            "run_id": run_id,
            "algorithm": args.algorithm,
            "messages_sent": messages_sent,
            "sent_ns": time.time_ns(),
        })

    finally:
        stop_event.set()
        if recovery_thread is not None:
            recovery_thread.join(timeout=6)
        client.disconnect()
        client.loop_stop()

    # --------------------------------------------------------
    # CALCULATE FINAL RESULTS
    # --------------------------------------------------------

    reduction = 100 * (1 - messages_sent / count)
    avg_schedule = float(np.mean(lags_ms)) if lags_ms else None
    max_schedule = float(np.max(lags_ms)) if lags_ms else None
    summary = {
        "run_id": run_id,
        "algorithm": args.algorithm,
        "mode": "pid_adaptive" if args.pid else "fixed",
        "blockwise_fixed": bool(args.blockwise_fixed),
        "sensor_id": args.sensor,
        "feature": args.feature,
        "original_readings": count,
        "messages_published": messages_sent,
        "data_reduction_pct": reduction,
        "telemetry_payload_bytes": payload_bytes,
        "replay_duration_seconds": replay_duration,
        "average_publish_rate_messages_per_sec": messages_sent / replay_duration,
        "application_payload_rate_bytes_per_sec": payload_bytes / replay_duration,
        "average_scheduling_delay_ms": avg_schedule,
        "maximum_scheduling_delay_ms": max_schedule,
        "initial_retention_pct": args.keep * 100,
        "pid_target_delay_ms": args.pid_target_delay_ms if args.pid else None,
        "pid_chunk_samples": args.pid_chunk_samples if (args.pid or args.blockwise_fixed) else None,
        "synced_recovery": bool(args.synced_recovery),
        "network_recovery_sample": args.recovery_sample if args.synced_recovery else None,
        "network_recovery_success": network_event.get("success") if network_event else None,
        "network_recovery_elapsed_s": (
            network_event.get("delete_confirmed_elapsed_s") if network_event else None
        ),
    }
    save_results(run_id, summary, pid_history, network_event)

    # --------------------------------------------------------
    # DISPLAY PUBLISHER SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MQTT PUBLISHER SUMMARY")
    print("=" * 60)
    print("Run ID:", run_id)
    print("Algorithm:", args.algorithm)
    print("Mode:", summary["mode"])
    print("Original readings:", count)
    print("Messages published:", messages_sent)
    print(f"Data reduction: {reduction:.2f}%")
    print("Telemetry payload bytes:", payload_bytes)
    print(f"Replay duration: {replay_duration:.3f} seconds")
    print(f"Average publish rate: {messages_sent / replay_duration:.2f} messages/sec")
    print(f"Application payload rate: {payload_bytes / replay_duration:.2f} bytes/sec")
    if avg_schedule is not None:
        print(f"Average scheduling delay: {avg_schedule:.3f} ms")
        print(f"Maximum scheduling delay: {max_schedule:.3f} ms")
    if network_event is not None:
        print("Network transition successful:", network_event["success"])
        print("Target transition:", f"{network_event['target_transition_elapsed_s']:.3f}s")
        if network_event.get("delete_confirmed_elapsed_s") is not None:
            print(
                "Actual transition:",
                f"{network_event['delete_confirmed_elapsed_s']:.3f}s",
            )
    print("=" * 60)
    if network_event is not None and not network_event["success"]:
        raise RuntimeError(
            "Network recovery failed or was not confirmed; results are NOT a valid "
            "congestion-to-recovery trial. See network_events.json."
        )
    print("\nMQTT publishing completed.")


# ============================================================
# 10. COMMAND-LINE ARGUMENTS
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Intel Lab MQTT producer with fixed/PID retention and synced recovery"
    )
    parser.add_argument("--sensor", type=int, default=1)
    parser.add_argument(
        "--feature",
        choices=["temperature", "humidity", "light", "voltage"],
        default="temperature",
    )
    parser.add_argument("--data", default="data/data.txt.gz")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--algorithm", choices=["raw"] + list(METHODS), default="raw")
    parser.add_argument("--keep", type=float, default=0.2)
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--hp-alpha", type=float, default=0.1)
    parser.add_argument("--pe-weight", type=float, default=0.5)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--pid", action="store_true")
    parser.add_argument("--pid-chunk-samples", type=int, default=20)
    parser.add_argument("--pid-feedback-wait", type=float, default=0.75)
    parser.add_argument("--pid-target-delay-ms", type=float, default=100.0)
    parser.add_argument("--pid-min-keep", type=float, default=0.10)
    parser.add_argument("--pid-max-keep", type=float, default=0.50)
    parser.add_argument("--pid-kp", type=float, default=0.0015)
    parser.add_argument("--pid-ki", type=float, default=0.00003)
    parser.add_argument("--pid-kd", type=float, default=0.00001)
    parser.add_argument("--pid-smoothing", type=float, default=0.4)
    parser.add_argument(
        "--blockwise-fixed", action="store_true",
        help="Select each fixed-retention block using the same 20-reading blocks as PID",
    )
    parser.add_argument(
        "--synced-recovery", action="store_true",
        help="Automatically apply 1 KB/s upstream bandwidth, then remove at --recovery-sample",
    )
    parser.add_argument("--recovery-sample", type=int, default=150)
    args = parser.parse_args()

    if not 1 <= args.sensor <= 54:
        parser.error("--sensor must be between 1 and 54")
    if args.max_samples < 100:
        parser.error("--max-samples must be at least 100")
    if not 0 < args.keep <= 1:
        parser.error("--keep must be between 0 and 1")
    if args.window < 6 or args.chunk < 8:
        parser.error("--window must be >=6 and --chunk must be >=8")
    if not (0 < args.hp_alpha <= 1 and 0 <= args.pe_weight <= 1):
        parser.error("Invalid algorithm parameters")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.pid_chunk_samples < 8:
        parser.error("--pid-chunk-samples must be >=8")
    if args.pid_feedback_wait < 0:
        parser.error("--pid-feedback-wait cannot be negative")
    if args.pid:
        if args.algorithm != "hybrid":
            parser.error("Initial PID experiments require --algorithm hybrid")
        if args.blockwise_fixed:
            parser.error("--blockwise-fixed is for the fixed baseline, not --pid")
        if not (0 < args.pid_min_keep <= args.keep <= args.pid_max_keep <= 1):
            parser.error("Require 0 < pid-min-keep <= keep <= pid-max-keep <= 1")
    if args.synced_recovery and not 0 < args.recovery_sample < args.max_samples:
        parser.error("--recovery-sample must be between 1 and max-samples - 1")
    start_publishing(args)


if __name__ == "__main__":
    main()