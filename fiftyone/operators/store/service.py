"""
FiftyOne execution store service.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import logging
import traceback
from fiftyone.factory.repo_factory import RepositoryFactory
from fiftyone.operators.store.permissions import StorePermissions

logger = logging.getLogger(__name__)


class ExecutionStoreService(object):
    """Service for managing execution store operations."""

    def __init__(self, repo=None):
        if repo is None:
            repo = RepositoryFactory.execution_store_repo()

        self._repo = repo

    def create_store(self, store_name, permissions=None):
        """Creates a new store with the specified name and permissions.

        Args:
            store_name: the name of the store
            permissions (None): an optional permissions dict

        Returns:
            a :class:`fiftyone.store.models.StoreDocument`
        """
        return self._repo.create_store(
            store_name=store_name,
            permissions=permissions or StorePermissions.default(),
        )

    def set_key(self, store_name, key, value, ttl=None):
        """Sets the value of a key in the specified store.

        Args:
            store_name: the name of the store
            key: the key to set
            value: the value to set
            ttl (None): an optional TTL in milliseconds

        Returns:
            a :class:`fiftyone.store.models.KeyDocument`
        """
        return self._repo.set_key(
            store_name=store_name, key=key, value=value, ttl=ttl
        )

    def get_key(self, store_name, key):
        """Retrieves the value of a key from the specified store.

        Args:
            store_name: the name of the store
            key: the key to retrieve

        Returns:
            a :class:`fiftyone.store.models.KeyDocument`
        """
        return self._repo.get_key(store_name=store_name, key=key)

    def delete_key(self, store_name, key):
        """Deletes the specified key from the store.

        Args:
            store_name: the name of the store
            key: the key to delete

        Returns:
            a :class:`fiftyone.store.models.KeyDocument`
        """
        return self._repo.delete_key(store_name=store_name, key=key)

    def update_ttl(self, store_name, key, new_ttl):
        """Updates the TTL of the specified key in the store.

        Args:
            store_name: the name of the store
            key: the key to update the TTL for
            new_ttl: the new TTL in milliseconds

        Returns:
            a :class:`fiftyone.store.models.KeyDocument`
        """
        return self._repo.update_ttl(
            store_name=store_name, key=key, ttl=new_ttl
        )

    def set_permissions(self, store_name, permissions):
        """Sets the permissions for the specified store.

        Args:
            store_name: the name of the store
            permissions: a permissions object

        Returns:
            a :class:`fiftyone.store.models.StoreDocument`
        """
        return self._repo.update_permissions(
            store_name=store_name, permissions=permissions
        )

    def list_stores(self, search=None, **kwargs):
        """Lists all stores matching the given criteria.

        Args:
            search (None): optional search term dict

        Returns:
            a list of :class:`fiftyone.store.models.StoreDocument`
        """
        return self._repo.list_stores(search=search, **kwargs)

    def delete_store(self, store_name):
        """Deletes the specified store.

        Args:
            store_name: the name of the store

        Returns:
            a :class:`fiftyone.store.models.StoreDocument`
        """
        return self._repo.delete_store(store_name=store_name)
