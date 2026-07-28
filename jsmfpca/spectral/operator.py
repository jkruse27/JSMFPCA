import numpy as np


class ObservationOperator:
    def build(self, hours, basis, model):
        X = basis.design_matrix(hours)
        n_modes = model.n_components_
        H = np.kron(np.eye(n_modes), X)

        return H

    def response_vector(self, subject, model):
        return subject.centered.reshape(-1, order="F")
