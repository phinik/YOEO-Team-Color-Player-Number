import numpy as np


class EntropyMap:
    """
    Similar class than Metric, except that it is used to track the entropy
    with a confusion matrix.
    """

    def __init__(self, n_classes):
        self._n_classes = n_classes
        self._entropy_map = np.zeros(shape=(n_classes, n_classes))

    def __add__(self, other: 'EntropyMap'):
        assert type(other) == EntropyMap, "cannot add other than EntropyMap"
        assert other._n_classes == self._n_classes, "Dimensions mismatch"

        m = EntropyMap(self._n_classes)
        m._entropy_map = self._entropy_map + other._entropy_map

        return m

    def _tp(self, class_id: int) -> int:
        return self._entropy_map[class_id, class_id]

    def _fp(self, class_id: int) -> int:
        return np.sum(self._entropy_map[class_id, :]) - self._entropy_map[class_id, class_id]

    def _fn(self, class_id: int) -> int:
        return np.sum(self._entropy_map[:, class_id]) - self._entropy_map[class_id, class_id]

    def _tn(self, class_id: int) -> int:
        return np.sum(self._entropy_map) - self._tp(class_id) - self._fp(class_id) - self._fn(class_id)


    def update(self, entropy: float, pred: int, target: int) -> None:
        self._entropy_map[pred, target] += entropy

    def merge(self, metric: 'EntropyMap') -> None:
        self._entropy_map += metric._entropy_map

    def reset(self) -> None:
        self._entropy_map = np.zeros(shape=(self._n_classes, self._n_classes))

    def get_map(self) -> np.ndarray:
        return self._entropy_map
