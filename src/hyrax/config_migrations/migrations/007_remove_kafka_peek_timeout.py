"""Config migration: version 7 → version 8.

Removes ``[data_set.KafkaStreamDataset].peek_timeout``. Models are no longer
pre-flighted from a sample peeked off the stream, so there is no wait left for
the key to bound.
"""

from tomlkit.toml_document import TOMLDocument

from hyrax.config_migrations.migration_utils import migration_step


@migration_step(from_version=7)
def remove_kafka_peek_timeout(cfg: TOMLDocument) -> TOMLDocument:
    """Drop the obsolete ``peek_timeout`` key from ``[data_set.KafkaStreamDataset]``."""
    data_set = cfg.get("data_set")
    if not isinstance(data_set, dict):
        return cfg

    kafka = data_set.get("KafkaStreamDataset")
    if isinstance(kafka, dict):
        kafka.pop("peek_timeout", None)

    return cfg
