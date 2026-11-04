
"""
Intel IoT MQTT Consumer

Adapted from the Vicomtech consumer.

Receives downsampled Intel sensor readings via MQTT.

Measures:
- Messages received
- MQTT payload bytes received
- Application-level delivery delay
- Message reception rate
- Missing or duplicate messages

Saves:
- received.csv
- summary.json

MQTT topics:
q1 = sensor telemetry
q2 = acknowledgement
"""

import csv
import json
import sys
import time
import uuid

from pathlib import Path
from statistics import mean

import paho.mqtt.client as mqtt

from config import get_config


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_ROOT = PROJECT_ROOT / "mqtt_results"

config = get_config("mqtt")

BROKER = config.PARAMS["host"]
PORT = int(config.PARAMS["port"])

INPUT_TOPIC = "q1"
RESPONSE_TOPIC = "q2"

sessions = {}


# ============================================================
# 2. SAVE RESULTS
# ============================================================

def save_results(run_id, session):

    folder = OUTPUT_ROOT / run_id
    folder.mkdir(parents=True, exist_ok=True)

    records = session["records"]

    csv_file = folder / "received.csv"

    columns = [
        "run_id",
        "algorithm",
        "sensor_id",
        "feature",
        "message_id",
        "sample_index",
        "sensor_timestamp",
        "value",
        "sent_ns",
        "received_ns",
        "delivery_delay_ms",
        "payload_bytes"
    ]

    with csv_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(records)

    delays = [
        row["delivery_delay_ms"]
        for row in records
    ]

    received_count = len(records)

    expected_count = session.get(
        "expected_messages",
        received_count
    )

    total_bytes = sum(
        row["payload_bytes"]
        for row in records
    )

    if received_count > 1:

        elapsed = (
            records[-1]["received_ns"]
            - records[0]["received_ns"]
        ) / 1_000_000_000

    else:
        elapsed = 0

    reception_rate = (
        received_count / elapsed
        if elapsed > 0
        else 0
    )

    summary = {
        "run_id": run_id,
        "algorithm": session["algorithm"],
        "messages_expected": expected_count,
        "messages_received": received_count,
        "missing_messages": max(
            0,
            expected_count - received_count
        ),
        "duplicate_messages": (
            session["duplicates"]
        ),
        "payload_bytes_received": total_bytes,
        "average_delivery_delay_ms": (
            mean(delays) if delays else None
        ),
        "minimum_delivery_delay_ms": (
            min(delays) if delays else None
        ),
        "maximum_delivery_delay_ms": (
            max(delays) if delays else None
        ),
        "reception_rate_messages_per_second": (
            reception_rate
        )
    }

    summary_file = folder / "summary.json"

    with summary_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )

    print("\n" + "=" * 55)
    print("MQTT EXPERIMENT COMPLETED")
    print("=" * 55)

    print("Algorithm:", session["algorithm"])
    print("Messages expected:", expected_count)
    print("Messages received:", received_count)
    print("Missing messages:", summary["missing_messages"])
    print("Duplicate messages:", session["duplicates"])

    print(
        "Telemetry payload bytes:",
        total_bytes
    )

    if delays:
        print(
            "Average delivery delay:",
            f"{mean(delays):.3f} ms"
        )

    print(
        "Reception rate:",
        f"{reception_rate:.2f} msg/s"
    )

    print("\nResults saved to:")
    print(folder.resolve())
    print("=" * 55)


# ============================================================
# 3. MQTT CONNECTION CALLBACK
# ============================================================

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

    print("\nConnected to MQTT broker.")
    print("Broker:", BROKER)
    print("Port:", PORT)

    client.subscribe(
        INPUT_TOPIC,
        qos=1
    )

    print(
        f"Subscribed to '{INPUT_TOPIC}'"
    )

    print("Waiting for Intel sensor readings...\n")


# ============================================================
# 4. MQTT MESSAGE CALLBACK
# ============================================================

def on_message(client, userdata, message):

    received_ns = time.time_ns()

    try:

        payload = json.loads(
            message.payload.decode("utf-8")
        )

        run_id = payload.get("run_id")

        if not run_id:
            print("Ignoring message without run_id.")
            return

        # The producer generates a hexadecimal run ID.
        if (
            len(run_id) > 64
            or not all(
                char in "0123456789abcdef"
                for char in run_id.lower()
            )
        ):
            print("Ignoring invalid run_id.")
            return

        # ====================================================
        # END OF EXPERIMENT
        # ====================================================

        if payload.get("id") == -1:

            if run_id not in sessions:

                print(
                    "End marker received without telemetry."
                )

                return

            session = sessions.pop(run_id)

            session["expected_messages"] = (
                payload.get(
                    "messages_sent",
                    len(session["records"])
                )
            )

            save_results(
                run_id,
                session
            )

            # Preserve the original q2 response topic.
            response = {
                "id": -1,
                "data": 1,
                "type": "experiment_end_ack",
                "run_id": run_id
            }

            client.publish(
                RESPONSE_TOPIC,
                json.dumps(response),
                qos=1
            )

            return

        # ====================================================
        # START OR CONTINUE A SESSION
        # ====================================================

        if run_id not in sessions:

            sessions[run_id] = {
                "algorithm": payload["algorithm"],
                "records": [],
                "seen_ids": set(),
                "duplicates": 0
            }

            print(
                f"\nNew experiment: {run_id}"
            )

            print(
                "Algorithm:",
                payload["algorithm"]
            )

        session = sessions[run_id]

        message_id = int(payload["id"])

        # Avoid counting repeated QoS 1 deliveries.
        if message_id in session["seen_ids"]:

            session["duplicates"] += 1
            return

        session["seen_ids"].add(message_id)

        # ====================================================
        # CALCULATE DELIVERY DELAY
        # ====================================================

        sent_ns = int(payload["sent_ns"])

        delay_ms = (
            received_ns - sent_ns
        ) / 1_000_000

        record = {
            "run_id": run_id,
            "algorithm": payload["algorithm"],
            "sensor_id": payload["sensor_id"],
            "feature": payload["feature"],
            "message_id": message_id,
            "sample_index": payload["sample_index"],
            "sensor_timestamp": (
                payload["sensor_timestamp"]
            ),
            "value": payload["data"],
            "sent_ns": sent_ns,
            "received_ns": received_ns,
            "delivery_delay_ms": delay_ms,
            "payload_bytes": len(message.payload)
        }

        session["records"].append(record)

        print(
            f"Received {message_id:03d} | "
            f"{payload['algorithm']} | "
            f"Value: {float(payload['data']):.3f} | "
            f"Delay: {delay_ms:.3f} ms"
        )

        # ====================================================
        # SEND RESPONSE TO q2
        # ====================================================

        response = {
            "id": message_id,
            "data": payload["data"],
            "run_id": run_id,
            "type": "ack",
            "received_ns": received_ns,
            "responded_ns": time.time_ns()
        }

        client.publish(
            RESPONSE_TOPIC,
            json.dumps(response),
            qos=1
        )

    except (
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError
    ) as error:

        print(
            "Invalid MQTT message:",
            error
        )


# ============================================================
# 5. START CONSUMER
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=(
            "intel_cloud_"
            + uuid.uuid4().hex[:8]
        )
    )

    client.on_connect = on_connect
    client.on_message = on_message

    print("Starting Intel MQTT consumer...")
    print(f"Connecting to {BROKER}:{PORT}")

    try:

        client.connect(
            BROKER,
            PORT,
            keepalive=60
        )

        client.loop_forever()

    except KeyboardInterrupt:

        print("\nConsumer stopped.")

    except OSError as error:

        print("MQTT connection error:", error)
        sys.exit(1)

    finally:

        client.disconnect()


if __name__ == "__main__":
    main()