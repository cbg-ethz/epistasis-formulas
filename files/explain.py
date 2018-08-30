import numpy as np

import circuits
import fourier
import slicing


class Explanator(object):

    def __init__(self, species=5):
        self.species = species
        self._w = np.arange(2**self.species)
        self._tp = slicing.TensorProjector(2, species)
        self._coordinates = dict()
        self._circuits = None

    def explain_in_binary(self, tag):
        return [
            np.binary_repr(t, width=self.species)
            for t in self.explain(tag)
        ]

    def explain(self, tag):
        pos = tag.find('_')
        if pos == -1:
            raise ValueError('Unable to find underscore delimiter in tag {0:s}'.format(tag))
        else:
            name = tag[:pos]
            setup = tag[(pos + 1):]

        if name == 'u':
            return self.explain_coordinate(setup)
        return self.explain_circuit(name, setup)

    def explain_coordinate(self, setup):
        idx_s = ''.join(
            '0' if c.islower() else '1'
            for c in setup
            if c not in '01'
        )
        order = len(idx_s)
        idx = int(idx_s, base=2)
        c = self._get_coordinate_by(order, idx)

        projection = self._project_according_to(setup.upper())
        return [p for c, p in zip(c, projection) if c != 0]

    def explain_circuit(self, name, setup):
        circuit_idx = self._circuit_name_to_idx(name)
        c = self._get_circuit_by(circuit_idx)

        projection = self._project_according_to(setup)
        return [p for c, p in zip(c, projection) if c != 0]

    def _get_coordinate_by(self, order, idx):
        if self._coordinates.get(order, None) is None:
            self._coordinates[order] = fourier.generate_full_fourier_matrix(order)

        return self._coordinates[order][idx]

    def _get_circuit_by(self, idx):
        if self._circuits is None:
            self._circuits = circuits.gen_circuits_3()
        return self._circuits[idx]

    def _project_according_to(self, setup):
        fixed_dims = [(d, int(s)) for d, s in enumerate(setup) if s in '01']
        return self._tp.project_vector(self._w, fixed_dims).flatten()

    @staticmethod
    def _circuit_name_to_idx(name):
        assert len(name) == 1, 'Circuit name of unexpected length'
        name = name.lower()
        assert 'a' <= name <= 't', 'Bad circuit name'
        return ord(name) - ord('a')
