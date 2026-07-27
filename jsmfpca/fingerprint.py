"""
Stage 3: construction of the interpretable subject fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectral.model import SpectralSubject


# ---------------------------------------------------------------------
# Final fingerprint
# ---------------------------------------------------------------------

@dataclass(slots=True)
class Fingerprint:
    """
    Low-dimensional subject representation.
    """

    subject_id: str | int

    vector: np.ndarray

    feature_names: list[str]

    @property
    def dimension(self):
        return self.vector.size

    def to_dict(self):

        return dict(
            zip(
                self.feature_names,
                self.vector,
            )
        )


# ---------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------

class FingerprintBuilder:

    """
    Builds the Stage 3 representation.

    Parameters
    ----------
    retained_components

        Dictionary

            harmonic -> number of retained components
    """

    def __init__(
        self,
        retained_components: dict[int, int],
    ):

        self.retained_components = retained_components

    # ---------------------------------------------------------

    def transform(
        self,
        subject: SpectralSubject,
    ) -> Fingerprint:

        features = []
        names = []

        #
        # Stage 1
        #

        for k, value in enumerate(subject.offsets, start=1):

            features.append(value)

            names.append(
                f"offset_mode_{k}"
            )

        #
        # Stage 2
        #

        coef = subject.rotated_coefficients

        n_harmonics = coef.shape[0] // 2

        for r in range(n_harmonics):

            keep = self.retained_components.get(
                r + 1,
                coef.shape[1],
            )

            cosine = coef[2 * r, :keep]

            sine = coef[2 * r + 1, :keep]

            for c, value in enumerate(cosine, start=1):

                features.append(value)

                names.append(
                    f"h{r+1}_component_{c}_cos"
                )

            for c, value in enumerate(sine, start=1):

                features.append(value)

                names.append(
                    f"h{r+1}_component_{c}_sin"
                )

        return Fingerprint(
            subject_id=subject.subject_id,
            vector=np.asarray(features),
            feature_names=names,
        )

    # ---------------------------------------------------------

    def transform_dataset(
        self,
        subjects,
    ):

        return [
            self.transform(subject)
            for subject in subjects
        ]
