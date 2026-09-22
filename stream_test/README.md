The files in this directory are here to help with testing streaming inference.

To use `producer.py` you'll need to have a docker container running like this:
```
docker run -d --name kafka -p 9092:9092 apache/kafka:3.7.0
```

To run `producer.py`:
```
python producer.py
```

The default connection parameters defined in producer should utilize the docker
container running kafka. 
If you need to change the connection parameters, you can do so in the `producer.py` file.



To reset the docker container:
```
docker kill kafka; docker rm kafka;  docker run -d --name kafka -p 9092:9092 apache/kafka:3.7.0
```
