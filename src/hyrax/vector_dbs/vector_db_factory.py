from typing import Union

from hyrax.vector_dbs.vector_db_interface import VectorDB


def vector_db_factory(config: dict, context: dict) -> Union[VectorDB, None]:
    """Factory method to create a database object"""

    # if the vector_db name is `False`, return None
    if not config["vector_db"]["name"]:
        return None

    vector_db_name = config["vector_db"]["name"]

    if vector_db_name == "chromadb":
        from hyrax.vector_dbs.chromadb_impl import ChromaDB

        return ChromaDB(config, context)
    elif vector_db_name == "qdrant":
        from hyrax.vector_dbs.qdrantdb_impl import QdrantDB

        return QdrantDB(config, context)
    else:
        return None
