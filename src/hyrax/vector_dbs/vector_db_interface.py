from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np

from hyrax.context import get_context


class VectorDB(ABC):
    """Interface for a vector database"""

    def __init__(self, config: dict | None = None):
        """
        .. py:method:: __init__

        Create a new instance of a `VectorDB` object.

        The directory the database lives in is read from the Hyrax run context,
        so callers do not need to pass it. Whoever creates a database is
        responsible for pointing the run context at the right directory first
        (see :func:`hyrax.context.update_context`).

        Parameters
        ----------
        config : dict, optional
            An instance of the runtime configuration, by default None
        """
        self.config = config if config else {}

        # Hold the context of the run that created us. Run contexts are per-run
        # objects that are released when their verb finishes, so this keeps
        # pointing at our own database directory however many verbs run later.
        # That matters because a database object outlives the run that created it
        # - database_connection hands one back to the user for interactive
        # querying - and subclasses read results_dir at query time.
        self.context = get_context()

    @property
    def results_dir(self) -> Path:
        """The directory this database lives in."""
        return self.context["results_dir"]

    @abstractmethod
    def connect(self):
        """Connect to an existing database"""
        pass

    @abstractmethod
    def create(self):
        """Create a new database"""
        pass

    @abstractmethod
    def insert(self, ids: list[Union[str, int]], vectors: list[np.ndarray]):
        """Insert a batch of vectors into the database.

        Parameters
        ----------
        ids : list[Union[str, int]]
            The ids to associate with the vectors
        vectors : list[np.ndarray]
            The vectors to insert into the database
        """
        pass

    @abstractmethod
    def search_by_id(self, id: Union[str, int], k: int = 1) -> dict[int, list[Union[str, int]]]:
        """Get the ids of the k nearest neighbors for a given id in the database.
        Should use the provided id to look up the vector, then call search_by_vector.

        Parameters
        ----------
        id : Union[str, int]
            The id of the vector in the database for which we want to find the
            k nearest neighbors
        k : int, optional
            The number of nearest neighbors to return, by default 1, return only
            the closest neighbor

        Returns
        -------
        dict[int, list[Union[str, int]]]
            Dictionary with input vector index as the key and the ids of the k
            nearest neighbors as the value.
        """
        pass

    @abstractmethod
    def search_by_vector(
        self, vectors: Union[np.ndarray, list[np.ndarray]], k: int = 1
    ) -> dict[int, list[Union[str, int]]]:
        """Get the ids of the k nearest neighbors for a given vector.

        Parameters
        ----------
        vectors : Union[np.array, list[np.ndarray]]
            The one or more vectors to use when searching for nearest neighbors
        k : int, optional
            The number of nearest neighbors to return, by default 1, return only
            the closest neighbor

        Returns
        -------
        dict[int, list[Union[str, int]]]
            Dictionary with input vector index as the key and the ids of the
            k nearest neighbors as the value.
        """
        pass

    @abstractmethod
    def get_by_id(self, ids: list[Union[str, int]]) -> dict[Union[str, int], list[float]]:
        """Retrieve the vectors associated with a list of ids.

        Parameters
        ----------
        ids : list[Union[str, int]]
            The ids of the vectors to retrieve.

        Returns
        -------
        dict[Union[str, int], list[float]]
            Dictionary with the ids as the keys and the vectors as the values.
        """
        pass
