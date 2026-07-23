The files in this directory are here to help with testing streaming inference.

To use `producer.py` you'll need to have a docker container running like this:
```
docker run -d --name kafka -p 9092:9092 apache/kafka:3.7.0
```

To run `producer.py`:
```
python producer.py
```

To run the dataset-backed producer:
```
python hyrax_dataset_producer.py --dataset cifar
python hyrax_dataset_producer.py --dataset hsc --data-dir /path/to/hsc/data
```

The dataset-backed producer emits flat JSON messages with `object_id`, `image`, and
dataset metadata. That shape is convenient for `KafkaStreamDataset` experiments
with `train_stream`.

The default connection parameters defined in producer should utilize the docker
container running kafka. 
If you need to change the connection parameters, you can do so in the `producer.py` file.



To reset the docker container:
```
docker kill kafka; docker rm kafka;  docker run -d --name kafka -p 9092:9092 apache/kafka:3.7.0
```
