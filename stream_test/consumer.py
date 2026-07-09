"""
Kafka consumer (for testing)
-----------------------------
Subscribes to a topic and prints every message as it arrives.
Starts from the beginning of the topic by default — useful for
re-reading all messages after a reset.

Usage:
    python consumer.py
    python consumer.py --topic my-topic --from-beginning
"""

import argparse
import json

from confluent_kafka import Consumer, KafkaError

DEFAULTS = dict(
    broker="localhost:9092",
    topic="my-topic",
    group="test-consumer-group",
    from_beginning=True,
)


def run(cfg: dict) -> None:
    offset_reset = "earliest" if cfg["from_beginning"] else "latest"

    consumer = Consumer(
        {
            "bootstrap.servers": cfg["broker"],
            "group.id": cfg["group"],
            "auto.offset.reset": offset_reset,
            "enable.auto.commit": True,
        }
    )

    consumer.subscribe([cfg["topic"]])
    print(f"[consumer] Subscribed to '{cfg['topic']}'  (offset_reset={offset_reset}, group={cfg['group']})")
    print("           Ctrl-C to stop.\n")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    print(f"[consumer] Reached end of partition {msg.partition()}")
                else:
                    print(f"[consumer] Error: {msg.error()}")
                continue

            payload = json.loads(msg.value().decode("utf-8"))
            print(
                f"offset={msg.offset():>6}  partition={msg.partition()}  "
                f"key={msg.key().decode()}  value={payload}"
            )

    except KeyboardInterrupt:
        print("\n[consumer] Interrupted — closing.")
    finally:
        consumer.close()


def parse_args() -> dict:
    p = argparse.ArgumentParser(description="Kafka consumer")
    p.add_argument("--broker", default=DEFAULTS["broker"])
    p.add_argument("--topic", default=DEFAULTS["topic"])
    p.add_argument("--group", default=DEFAULTS["group"])
    p.add_argument(
        "--from-beginning",
        action="store_true",
        default=DEFAULTS["from_beginning"],
        help="Read from offset 0 (default: True)",
    )
    p.add_argument("--latest", dest="from_beginning", action="store_false", help="Read only new messages")
    return vars(p.parse_args())


if __name__ == "__main__":
    run(parse_args())
