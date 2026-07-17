"""
Kafka dataset producer
----------------------
Emits messages in bursts with random delays between them, sampling images from
either HyraxCifarDataset or HSCDataset.

Install:
    pip install confluent-kafka torchvision astropy schwimmbad

Run a local Kafka broker (Docker):
    docker run -d --name kafka -p 9092:9092 apache/kafka:3.7.0

Usage:
    python hyrax_dataset_producer.py --dataset cifar
    python hyrax_dataset_producer.py --dataset hsc --data-dir /path/to/hsc/data
    python hyrax_dataset_producer.py --dataset hsc --data-dir /path/to/hsc/data \
        --topic train-stream --burst-min 3 --burst-max 8 --delay-min 5 --delay-max 15
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

from hyrax.datasets import HSCDataset, HyraxCifarDataset

DEFAULTS = dict(
    broker="localhost:9092",
    topic="my-topic",
    dataset="cifar",
    data_dir="data",
    burst_min=10,
    burst_max=40,
    delay_min=10,
    delay_max=15,
    num_bursts=3,
    use_training_data=True,
)


def ensure_topic(broker: str, topic: str, partitions: int = 1, replication: int = 1) -> None:
    """Create the topic if it doesn't already exist."""
    admin = AdminClient({"bootstrap.servers": broker})
    existing = admin.list_topics(timeout=5).topics
    if topic not in existing:
        fs = admin.create_topics([NewTopic(topic, num_partitions=partitions, replication_factor=replication)])
        for name, future in fs.items():
            try:
                future.result()
                print(f"[setup] Created topic '{name}'")
            except Exception as err:
                print(f"[setup] Topic '{name}' already exists or error: {err}")
    else:
        print(f"[setup] Topic '{topic}' already exists — reusing it.")


def delivery_report(err, msg) -> None:
    """Called once for each message produced to indicate delivery result."""
    if err:
        print(f"  [!] Delivery failed: {err}")
    else:
        print(f"  [✓] offset={msg.offset():>6}  key={msg.key().decode()}")


def _to_python(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def build_cifar_dataset(data_dir: str, use_training_data: bool) -> HyraxCifarDataset:
    """Build a CIFAR-backed Hyrax dataset from local config values."""
    config = {"data_set": {"HyraxCifarDataset": {"use_training_data": use_training_data}}}
    return HyraxCifarDataset(config, data_location=Path(data_dir).expanduser().resolve())


def build_hsc_dataset(data_dir: str) -> HSCDataset:
    """Build an HSC-backed Hyrax dataset from a local data directory."""
    config = {
        "data_set": {
            "crop_to": False,
            "filters": False,
            "filter_catalog": False,
            "use_cache": False,
            "transform": False,
            "object_id_column_name": False,
            "filter_column_name": False,
            "filename_column_name": False,
        },
    }
    return HSCDataset(config, data_location=Path(data_dir).expanduser().resolve())


def build_dataset(cfg: dict):
    """Build the configured dataset implementation."""
    dataset_name = cfg["dataset"]
    if dataset_name == "cifar":
        return build_cifar_dataset(cfg["data_dir"], cfg["use_training_data"])
    if dataset_name == "hsc":
        return build_hsc_dataset(cfg["data_dir"])
    raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected 'cifar' or 'hsc'.")


def make_message(dataset, dataset_name: str, index: int) -> dict:
    """Create a JSON-serializable payload for one dataset sample."""
    image = _to_python(dataset.get_image(index))
    payload = {
        "dataset": dataset_name,
        "dataset_class": dataset.__class__.__name__,
        "id": str(dataset.get_object_id(index)),
        "index": index,
        "image": image,
        "image_shape": list(np.asarray(image).shape),
    }

    if hasattr(dataset, "get_label"):
        payload["label"] = _to_python(dataset.get_label(index))

    return payload


def run(cfg: dict) -> None:
    """Run the Kafka producer loop using the configured dataset."""
    dataset = build_dataset(cfg)
    dataset_name = cfg["dataset"]

    if len(dataset) == 0:
        raise RuntimeError(f"Selected dataset '{dataset_name}' contains no samples.")

    ensure_topic(cfg["broker"], cfg["topic"])

    producer = Producer({"bootstrap.servers": cfg["broker"]})
    total_sent = 0
    burst_count = 0

    print(f"\n[producer] Starting — broker={cfg['broker']}  topic={cfg['topic']}")
    print(
        f"           dataset={dataset.__class__.__name__}  "
        f"bursts={'∞' if cfg['num_bursts'] == 0 else cfg['num_bursts']}  "
        f"msgs/burst={cfg['burst_min']}–{cfg['burst_max']}  "
        f"delay={cfg['delay_min']}–{cfg['delay_max']}s\n"
    )

    try:
        while cfg["num_bursts"] == 0 or burst_count < cfg["num_bursts"]:
            n = random.randint(cfg["burst_min"], cfg["burst_max"])
            burst_count += 1
            print(f"── Burst #{burst_count}  ({n} messages) ──────────────────────")

            for _ in range(n):
                index = random.randrange(len(dataset))
                payload = make_message(dataset, dataset_name, index)
                producer.produce(
                    topic=cfg["topic"],
                    key=payload["id"],
                    value=json.dumps(payload),
                    callback=delivery_report,
                )
                producer.poll(0)

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


def parse_args() -> dict:
    """Parse command-line arguments and return a configuration dictionary."""
    parser = argparse.ArgumentParser(description="Kafka producer backed by Hyrax datasets")
    parser.add_argument("--broker", default=DEFAULTS["broker"])
    parser.add_argument("--topic", default=DEFAULTS["topic"])
    parser.add_argument("--dataset", choices=("cifar", "hsc"), default=DEFAULTS["dataset"])
    parser.add_argument("--data-dir", default=DEFAULTS["data_dir"])
    parser.add_argument("--burst-min", type=int, default=DEFAULTS["burst_min"])
    parser.add_argument("--burst-max", type=int, default=DEFAULTS["burst_max"])
    parser.add_argument("--delay-min", type=float, default=DEFAULTS["delay_min"])
    parser.add_argument("--delay-max", type=float, default=DEFAULTS["delay_max"])
    parser.add_argument(
        "--num-bursts",
        type=int,
        default=DEFAULTS["num_bursts"],
        help="Number of bursts to emit (0 = run forever)",
    )
    parser.add_argument(
        "--use-training-data",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS["use_training_data"],
        help="Use the CIFAR training split when dataset=cifar.",
    )
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    run(parse_args())
