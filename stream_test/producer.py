"""
Kafka burst producer
--------------------
Emits messages in bursts with random delays between them.

Install:
    pip install confluent-kafka

Run a local Kafka broker (Docker):
    docker run -d --name kafka -p 9092:9092 apache/kafka:3.7.0

Reset the topic (delete + recreate):
    docker exec kafka /opt/kafka/bin/kafka-topics.sh \
        --bootstrap-server localhost:9092 --delete --topic my-topic
    docker exec kafka /opt/kafka/bin/kafka-topics.sh \
        --bootstrap-server localhost:9092 --create --topic my-topic

Usage:
    python producer.py
    python producer.py --topic alerts --broker localhost:9092 \
        --burst-min 3 --burst-max 8 --delay-min 5 --delay-max 15
"""

import argparse
import json
import random
import time
import uuid
from datetime import UTC, datetime

import numpy as np
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

# ── Config defaults ────────────────────────────────────────────────────────────

DEFAULTS = dict(
    broker="localhost:9092",
    topic="my-topic",
    burst_min=10,  # min messages per burst
    burst_max=40,  # max messages per burst
    delay_min=10,  # min seconds between bursts
    delay_max=15,  # max seconds between bursts
    num_bursts=3,  # 0 = run forever
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def ensure_topic(broker: str, topic: str, partitions: int = 1, replication: int = 1) -> None:
    """Create the topic if it doesn't already exist."""
    admin = AdminClient({"bootstrap.servers": broker})
    existing = admin.list_topics(timeout=5).topics
    if topic not in existing:
        fs = admin.create_topics([NewTopic(topic, num_partitions=partitions, replication_factor=replication)])
        for t, f in fs.items():
            try:
                f.result()
                print(f"[setup] Created topic '{t}'")
            except Exception as e:
                print(f"[setup] Topic '{t}' already exists or error: {e}")
    else:
        print(f"[setup] Topic '{topic}' already exists — reusing it.")


def delivery_report(err, msg) -> None:
    """Called once for each message produced to indicate delivery result."""
    if err:
        print(f"  [!] Delivery failed: {err}")
    else:
        print(f"  [✓] offset={msg.offset():>6}  key={msg.key().decode()}")


def make_message(index: int) -> dict:
    """Build a simple payload. Replace with your own schema."""
    return {
        "id": str(uuid.uuid4()),
        "index": index,
        "value": round(random.uniform(0, 100), 4),
        "image": np.random.rand(4, 4).tolist(),
        "label": np.random.randint(0, 9),
        "timestamp": datetime.now(UTC).isoformat(),
    }


# ── Main ───────────────────────────────────────────────────────────────────────


def run(cfg: dict) -> None:
    """Runs the producer

    Parameters
    ----------
    cfg : dict
        The configuration parameters.
    """
    ensure_topic(cfg["broker"], cfg["topic"])

    producer = Producer({"bootstrap.servers": cfg["broker"]})
    total_sent = 0
    burst_count = 0

    print(f"\n[producer] Starting — broker={cfg['broker']}  topic={cfg['topic']}")
    print(
        f"           bursts={'∞' if cfg['num_bursts'] == 0 else cfg['num_bursts']}  "
        f"msgs/burst={cfg['burst_min']}–{cfg['burst_max']}  "
        f"delay={cfg['delay_min']}–{cfg['delay_max']}s\n"
    )

    try:
        while cfg["num_bursts"] == 0 or burst_count < cfg["num_bursts"]:
            n = random.randint(cfg["burst_min"], cfg["burst_max"])
            burst_count += 1
            print(f"── Burst #{burst_count}  ({n} messages) ──────────────────────")

            for i in range(n):
                payload = make_message(total_sent + i)
                producer.produce(
                    topic=cfg["topic"],
                    key=payload["id"],
                    value=json.dumps(payload),
                    callback=delivery_report,
                )
                producer.poll(0)  # serve delivery callbacks without blocking

            producer.flush()
            total_sent += n

            if cfg["num_bursts"] != 0 and burst_count >= cfg["num_bursts"]:
                break

            delay = random.uniform(cfg["delay_min"], cfg["delay_max"])
            print(f"   → next burst in {delay:.1f}s  (total sent: {total_sent})\n")
            time.sleep(delay)

    except KeyboardInterrupt:
        print("\n[producer] Interrupted — flushing…")
        producer.flush()

    print(f"\n[producer] Done. Total messages sent: {total_sent}")


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> dict:
    """Parse command-line arguments and return a configuration dictionary."""
    p = argparse.ArgumentParser(description="Kafka burst producer")
    p.add_argument("--broker", default=DEFAULTS["broker"])
    p.add_argument("--topic", default=DEFAULTS["topic"])
    p.add_argument("--burst-min", type=int, default=DEFAULTS["burst_min"])
    p.add_argument("--burst-max", type=int, default=DEFAULTS["burst_max"])
    p.add_argument("--delay-min", type=float, default=DEFAULTS["delay_min"])
    p.add_argument("--delay-max", type=float, default=DEFAULTS["delay_max"])
    p.add_argument(
        "--num-bursts",
        type=int,
        default=DEFAULTS["num_bursts"],
        help="Number of bursts to emit (0 = run forever)",
    )
    args = p.parse_args()
    return vars(args)


if __name__ == "__main__":
    run(parse_args())
