import numpy as np
from abc import ABC, abstractmethod


class Coreset(ABC):
    """
    Abstract base class for coresets
    """
    def __init__(self, coreset_size):
        """
        Initialise the coreset
        :param coreset_size: Size of the coreset
        :type coreset_size: int
        """
        self.coreset_size = coreset_size
        self.data = []
        self.labels = []

    @abstractmethod
    def update(self, data, labels):
        pass

    @staticmethod
    def get_merged(coresets):
        """
        Get merged data and labels from the list of coresets
        :param coresets: list of coresets
        :type coresets: list(Coreset)
        :return: merged data and labels
        """
        merged_data, merged_labels = coresets[0].data, coresets[0].labels
        for i in range(1, len(coresets)):
            merged_data = np.vstack((merged_data, coresets[i].data))
            merged_labels = np.vstack((merged_labels, coresets[i].labels))
        return merged_data, merged_labels


class Random(Coreset):
    """
    Randomly select from (data, labels) and add to current coreset
    """
    def update(self, data, labels):
        idx = np.random.choice(data.shape[0], self.coreset_size, False)
        self.data.append(data[idx, :])
        self.labels.append(labels[idx, :])
        data = np.delete(data, idx, axis=0)
        labels = np.delete(labels, idx, axis=0)
        return data, labels


class KCenter(Coreset):
    """
    Select k centers from (data, labels) and add to current coreset
    """
    def update(self, data, labels):
        dists = np.full(data.shape[0], np.inf)
        current_id = 0

        # TODO: This looks horribly inefficient
        dists = self.update_distance(dists, data, current_id)
        idx = [current_id]

        for i in range(1, self.coreset_size):
            current_id = np.argmax(dists)
            dists = self.update_distance(dists, data, current_id)
            idx.append(current_id)

        self.data.append(data[idx, :])
        self.labels.append(labels[idx, :])
        data = np.delete(data, idx, axis=0)
        labels = np.delete(labels, idx, axis=0)
        return data, labels

    @staticmethod
    def update_distance(dists, data, current_id):
        for i in range(data.shape[0]):
            current_dist = np.linalg.norm(data[i, :] - data[current_id, :])
            dists[i] = np.minimum(current_dist, dists[i])
        return dists
