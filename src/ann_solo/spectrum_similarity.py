import math
import warnings
from typing import Optional

import numpy as np
import scipy.spatial.distance
import scipy.special
import scipy.stats

from ann_solo import spectrum


class SpectrumSimilarityCalculator:
    def __init__(
        self, ssm: spectrum.SpectrumSpectrumMatch, top: Optional[int] = None
    ):
        """
        Instantiate the `SpectrumSimilarityCalculator` to compute various spectrum
        similarities between the two spectra in the `SpectrumSpectrumMatch`.

        Parameters
        ----------
        ssm : spectrum.SpectrumSpectrumMatch
            The match between a query spectrum and a library spectrum.
        top: Optional[int] = None
            The number of library peaks with highest intensity to consider. If
            `None`, all peaks are used.
        """
        self.mz_query = ssm.query_spectrum.mz
        self.int_query = ssm.query_spectrum.intensity
        self.mz_library = ssm.library_spectrum.mz
        self.int_library = ssm.library_spectrum.intensity
        self._top = top
        self._recalculate_norm = False
        if len(ssm.peak_matches) > 0:
            self.matched_mz_query = self.mz_query[ssm.peak_matches[:, 0]]
            self.matched_int_query = self.int_query[ssm.peak_matches[:, 0]]
            self.matched_mz_library = self.mz_library[ssm.peak_matches[:, 1]]
            self.matched_int_library = self.int_library[ssm.peak_matches[:, 1]]
            self.unmatched_int_query = self.int_query[
                np.setdiff1d(
                    range(len(self.int_query)), ssm.peak_matches[:, 0], True
                )
            ]
            library_unmatched_i = np.setdiff1d(
                range(len(self.int_library)), ssm.peak_matches[:, 1], True
            )
            self.unmatched_int_library = self.int_library[library_unmatched_i]
            # Filter the peak matches by the `top` highest intensity peaks in
            # the library spectrum.
            if self._top is not None:
                library_top_i = np.argpartition(self.int_library, -top)[-top:]
                mask = np.isin(
                    ssm.peak_matches[:, 1], library_top_i, assume_unique=True
                )
                # No matches remaining that include any of the `top` highest
                # intensity peaks.
                if not mask.any():
                    self.matched_mz_query = None
                    self.matched_int_query = None
                    self.matched_mz_library = None
                    self.matched_int_library = None
                else:
                    self.matched_mz_query = self.matched_mz_query[mask]
                    self.matched_int_query = self.matched_int_query[mask]
                    self.matched_mz_library = self.matched_mz_library[mask]
                    self.matched_int_library = self.matched_int_library[mask]
                # Also restrict the unmatched library peaks to the `top`
                # highest intensity peaks.
                mask_unmatched = np.isin(
                    library_unmatched_i, library_top_i, assume_unique=True
                )
                self.unmatched_int_library = self.unmatched_int_library[
                    mask_unmatched
                ]
                self._recalculate_norm = True
        else:
            self.matched_mz_query, self.matched_int_query = None, None
            self.matched_mz_library, self.matched_int_library = None, None

    def cosine(self) -> float:
        """
        Get the cosine similarity.

        For the original description, see:
        Bittremieux, W., Meysman, P., Noble, W. S. & Laukens, K. Fast open
        modification spectral library searching through approximate nearest
        neighbor indexing. Journal of Proteome Research 17, 3463–3474 (2018).

        Returns
        -------
        float
            The cosine similarity between the two spectra.
        """
        if self.matched_int_query is not None:
            if self._recalculate_norm:
                norm = np.linalg.norm(self.matched_int_query) * np.linalg.norm(
                    self.matched_int_library
                )
            else:
                norm = 1.0
            return (
                np.dot(self.matched_int_query, self.matched_int_library) / norm
            )
        else:
            return 0.0

    def n_matched_peaks(self) -> int:
        """
        Get the number of shared peaks.

        Returns
        -------
        int
            The number of matching peaks between the two spectra.
        """
        if self.matched_mz_query is not None:
            return len(self.matched_mz_query)
        else:
            return 0

    def frac_n_peaks_query(self) -> float:
        """
        Get the number of shared peaks as a fraction of the number of peaks in
        the query spectrum.

        Returns
        -------
        float
            The fraction of shared peaks in the query spectrum.
        """
        if self._top is not None:
            raise NotImplementedError(
                "The fraction of shared query peaks is not defined when "
                "filtering by the top intensity library peaks"
            )
        elif self.matched_mz_query is not None:
            return len(self.matched_mz_query) / len(self.mz_query)
        else:
            return 0.0

    def frac_n_peaks_library(self) -> float:
        """
        Get the number of shared peaks as a fraction of the number of peaks in
        the library spectrum.

        Returns
        -------
        float
            The fraction of shared peaks in the library spectrum.
        """
        if self.matched_int_library is not None:
            if self._top is None:
                n_peaks = len(self.mz_library)
            else:
                n_peaks = len(self.matched_int_library) + len(
                    self.unmatched_int_library
                )
            return len(self.matched_int_library) / n_peaks
        else:
            return 0.0

    def frac_intensity_query(self) -> float:
        """
        Get the fraction of explained intensity in the query spectrum.

        Returns
        -------
        float
            The fraction of explained intensity in the query spectrum.
        """
        if self._top:
            raise NotImplementedError(
                "The fraction of explained query intensity is not defined when"
                " filtering by the top intensity library peaks"
            )
        elif self.matched_int_query is not None:
            return self.matched_int_query.sum() / self.int_query.sum()
        else:
            return 0.0

    def frac_intensity_library(self) -> float:
        """
        Get the fraction of explained intensity in the library spectrum.

        Returns
        -------
        float
            The fraction of explained intensity in the library spectrum.
        """
        if self.matched_int_library is not None:
            if self._top is None:
                total_int = self.int_library.sum()
            else:
                total_int = (
                    self.matched_int_library.sum()
                    + self.unmatched_int_library.sum()
                )
            return self.matched_int_library.sum() / total_int
        else:
            return 0.0

    def mean_squared_error(self, axis: str) -> float:
        """
        Get the mean squared error (MSE) of peak matches.

        Parameters
        ----------
        axis : str
            Calculate the MSE between the m/z values ("mz") or intensity values
            ("intensity") of the matched peaks.

        Returns
        -------
        float
            The MSE between the m/z or intensity values of the matched peaks in
            the two spectra.

        Raises
        ------
        ValueError
            If the specified axis is not "mz" or "intensity".
        """
        if axis == "mz":
            arr1, arr2 = self.matched_mz_query, self.matched_mz_library
        elif axis == "intensity":
            arr1, arr2 = self.matched_int_query, self.matched_int_library
        else:
            raise ValueError("Unknown axis specified")
        if arr1 is not None and arr2 is not None:
            return ((arr1 - arr2) ** 2).sum() / len(arr1)
        else:
            return np.inf

    def spectral_contrast_angle(self) -> float:
        """
        Get the spectral contrast angle.

        For the original description, see:
        Toprak, U. H. et al. Conserved peptide fragmentation as a benchmarking
        tool for mass spectrometers and a discriminating feature for targeted
        proteomics. Molecular & Cellular Proteomics 13, 2056–2071 (2014).

        Returns
        -------
        float
            The spectral contrast angle between the two spectra.
        """
        return 1.0 - 2 * np.arccos(np.clip(self.cosine(), 0.0, 1.0)) / np.pi

    def hypergeometric_score(
        self, min_mz: float, max_mz: float, fragment_mz_tol: float
    ) -> float:
        """
        Get the hypergeometric score of peak matches.

        The hypergeometric score measures the probability of obtaining more
        than the observed number of peak matches by random chance, which
        follows a hypergeometric distribution.

        For the original description, see:
        Dasari, S. et al. Pepitome: Evaluating improved spectral library search
        for identification complementarity and quality assessment. Journal of
        Proteome Research 11, 1686–1695 (2012).

        Parameters
        ----------
        min_mz : float
            The minimum fragment m/z that is considered.
        max_mz : float
            The maximum fragment m/z that is considered.
        fragment_mz_tol : float
            The fragment m/z tolerance (in Da).

        Returns
        -------
        float
            The negative logarithm of the hypergeometric score of peak matches
            between the two spectra.
        """
        if self._top is not None:
            if self.matched_int_library is not None:
                n_library_peaks = len(self.matched_int_library) + len(
                    self.unmatched_int_library
                )
            else:
                n_library_peaks = self._top
        else:
            n_library_peaks = len(self.int_library)
        n_matched_peaks = (
            len(self.matched_mz_library)
            if self.matched_mz_library is not None
            else 0
        )
        n_peak_bins, _, _ = spectrum.get_dim(min_mz, max_mz, fragment_mz_tol)
        with np.errstate(divide="ignore"):
            hgt_prob = 0
            for i in range(n_matched_peaks + 1, n_library_peaks + 1):
                hgt_prob += (
                    scipy.special.comb(n_library_peaks, i)
                    * scipy.special.comb(
                        n_peak_bins - n_library_peaks, n_library_peaks - i
                    )
                ) / scipy.special.comb(n_peak_bins, n_library_peaks)
            # Guard against infinity for identical spectra.
            return min(-np.log(hgt_prob), 100.0)

    def kendalltau(self) -> float:
        """
        Get the Kendall-Tau score of peak matches.

        The Kendall-Tau score measures the correspondence between the intensity
        ranks of the set of peaks matched between spectra.

        For the original description, see:
        Dasari, S. et al. Pepitome: Evaluating improved spectral library search
        for identification complementarity and quality assessment. Journal of
        Proteome Research 11, 1686–1695 (2012).

        Returns
        -------
        float
            The negative logarithm of the Kendall-Tau p-value of peak matches
            between the two spectra.
        """
        pvalue = scipy.stats.kendalltau(
            self.matched_int_query, self.matched_int_library
        )[1]
        # The Kendall Tau correlation is undefined for constant input
        # arrays (in the degenerate case consisting of only 0 or 1 elements).
        return -np.log(pvalue) if not np.isnan(pvalue) else 0.0

    def ms_for_id_v1(self) -> float:
        """
        Get the MSforID (v1) similarity.

        For the original description, see:
        Pavlic, M., Libiseller, K. & Oberacher, H. Combined use of ESI–QqTOF-MS
        and ESI–QqTOF-MS/MS with mass-spectral library search for qualitative
        analysis of drugs. Analytical and Bioanalytical Chemistry 386, 69–82
        (2006).

        Returns
        -------
        float
            The MSforID (v1) similarity between the two spectra.
        """
        if self.matched_int_query is not None:
            if self._top is None:
                n_peaks_query = len(self.mz_query)
                n_peaks_library = len(self.mz_library)
            else:
                n_peaks_query = n_peaks_library = self._top
            # Guard against extreme values for identical spectra.
            return min(
                len(self.matched_int_query) ** 4
                / (
                    n_peaks_query
                    * n_peaks_library
                    * max(
                        np.abs(
                            self.matched_int_query - self.matched_int_library
                        ).sum(),
                        np.finfo(float).eps,
                    )
                    ** 0.25
                ),
                1000.0,
            )
        else:
            return 0.0

    def ms_for_id_v2(self) -> float:
        """
        Get the MSforID (v2) similarity.

        For the original description, see:
        Oberacher, H. et al. On the inter-instrument and the inter-laboratory
        transferability of a tandem mass spectral reference library: 2.
        Optimization and characterization of the search algorithm: About an
        advanced search algorithm for tandem mass spectral reference libraries.
        Journal of Mass Spectrometry 44, 494–502 (2009).

        Returns
        -------
        float
            The MSforID (v2) similarity between the two spectra.
        """
        if self._top is not None:
            raise NotImplementedError(
                "The MSforID (v2) similarity is not defined when filtering"
                " by the top intensity library peaks"
            )
        elif self.matched_int_query is not None:
            return (
                len(self.matched_int_query) ** 4
                * (self.int_query.sum() + 2 * self.int_library.sum()) ** 1.25
            ) / (
                (len(self.mz_query) + 2 * len(self.mz_library)) ** 2
                + np.abs(
                    self.matched_int_query - self.matched_int_library
                ).sum()
                + np.abs(self.matched_mz_query - self.matched_mz_library).sum()
            )
        else:
            return 0.0

    def manhattan(self) -> float:
        """
        Get the Manhattan distance.

        Returns
        -------
        float
            The Manhattan distance between the two spectra.
        """
        if self._top is not None:
            raise NotImplementedError(
                "The Manhattan distance is not defined when filtering by the "
                "top intensity library peaks"
            )
        # Distance between the intensities of matching peaks, as well as the
        # unmatched intensities in the query and library spectrum.
        elif self.matched_int_query is not None:
            return (
                np.abs(self.matched_int_query - self.matched_int_library).sum()
                + self.unmatched_int_query.sum()
                + self.unmatched_int_library.sum()
            )
        else:
            return np.inf

    def euclidean(self) -> float:
        """
        Get the Euclidean distance.

        Returns
        -------
        float
            The Euclidean distance between the two spectra.
        """
        if self._top is not None:
            raise NotImplementedError(
                "The Euclidean distance is not defined when filtering by the "
                "top intensity library peaks"
            )
        # Distance between the intensities of matching peaks, as well as the
        # unmatched intensities in the query and library spectrum.
        elif self.matched_int_query is not None:
            return np.sqrt(
                (
                    (self.matched_int_query - self.matched_int_library) ** 2
                ).sum()
                + (self.unmatched_int_query**2).sum()
                + (self.unmatched_int_library**2).sum()
            )
        else:
            return np.inf

    def chebyshev(self) -> float:
        """
        Get the Chebyshev distance.

        Returns
        -------
        float
            The Chebyshev distance between the two spectra.
        """
        if self._top is not None:
            raise NotImplementedError(
                "The Chebyshev distance is not defined when filtering by the "
                "top intensity library peaks"
            )
        # Distance between the intensities of matching peaks, as well as the
        # unmatched intensities in the query and library spectrum.
        elif self.matched_int_query is not None:
            return max(
                np.abs(
                    self.matched_int_query - self.matched_int_library
                ).max(),
                self.unmatched_int_query.max()
                if len(self.unmatched_int_query) > 0
                else 0.0,
                self.unmatched_int_library.max()
                if len(self.unmatched_int_library) > 0
                else 0.0,
            )
        else:
            return np.inf

    def pearsonr(self) -> float:
        """
        Get the Pearson correlation between peak matches.

        Returns
        -------
        float
            The Pearson correlation of peak matches between the two spectra.
        """
        if self.matched_int_query is not None:
            int_query = [
                *self.matched_int_query,
                *np.zeros_like(self.unmatched_int_library),
            ]
            int_library = [
                *self.matched_int_library,
                *self.unmatched_int_library,
            ]
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore", scipy.stats.PearsonRConstantInputWarning
                )
                corr = scipy.stats.pearsonr(int_query, int_library)[0]
            return corr if not np.isnan(corr) else 0.0
        else:
            return 0.0

    def spearmanr(self) -> float:
        """
        Get the Spearman correlation between peak matches.

        Returns
        -------
        float
            The Spearman correlation of peak matches between the two spectra.
        """
        if self.matched_int_query is not None:
            int_query = [
                *self.matched_int_query,
                *np.zeros_like(self.unmatched_int_library),
            ]
            int_library = [
                *self.matched_int_library,
                *self.unmatched_int_library,
            ]
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore", scipy.stats.SpearmanRConstantInputWarning
                )
                corr = scipy.stats.spearmanr(int_query, int_library)[0]
            return corr if not np.isnan(corr) else 0.0
        else:
            return 0.0

    def braycurtis(self) -> float:
        """
        Get the Bray-Curtis distance.

        Returns
        -------
        float
            The Bray-Curtis distance between the two spectra.
        """
        if self._top is not None:
            raise NotImplementedError(
                "The Bray-Curtis distance is not defined when filtering by "
                "the top intensity library peaks"
            )
        elif self.matched_int_query is not None:
            unmatched_int_query_sum = self.unmatched_int_query.sum()
            unmatched_int_library_sum = self.unmatched_int_library.sum()
            return (
                np.abs(self.matched_int_query - self.matched_int_library).sum()
                + unmatched_int_query_sum
                + unmatched_int_library_sum
            ) / (
                np.abs(self.matched_int_query + self.matched_int_library).sum()
                + unmatched_int_query_sum
                + unmatched_int_library_sum
            )
        else:
            return 1.0

    def canberra(self) -> float:
        """
        Get the Canberra distance.

        Returns
        -------
        float
            The Canberra distance between the two spectra.
        """
        if self._top is not None:
            raise NotImplementedError(
                "The Canberra distance is not defined when filtering by the "
                "top intensity library peaks"
            )
        elif self.matched_int_query is not None:
            return (
                np.nan_to_num(
                    np.abs(self.matched_int_query - self.matched_int_library)
                    / (self.matched_int_query + self.matched_int_library),
                    copy=False,
                ).sum()
                + np.count_nonzero(self.unmatched_int_query)
                + np.count_nonzero(self.unmatched_int_library)
            )
        else:
            return np.inf

    def ruzicka(self) -> float:
        """
        Get the Ruzicka similarity.

        Returns
        -------
        float
            The Ruzicka similarity between the two spectra.
        """
        if self._top is not None:
            raise NotImplementedError(
                "The Ruzicka distance is not defined when filtering by the "
                "top intensity library peaks"
            )
        elif self.matched_int_query is not None:
            return np.sum(
                np.minimum(self.matched_int_query, self.matched_int_library)
            ) / (
                np.maximum(
                    self.matched_int_query, self.matched_int_library
                ).sum()
                + self.unmatched_int_query.sum()
                + self.unmatched_int_library.sum()
            )
        else:
            return 0.0

    def scribe_fragment_acc(self) -> float:
        """
        Get the Scribe fragmentation accuracy.

        For the original description, see:
        Searle, B. C. et al. Scribe: next-generation library searching for DDA
        experiments. ASMS 2022.

        Returns
        -------
        float
            The Scribe fragmentation accuracy between the two spectra.
        """
        if self.matched_int_query is not None:
            denominator = (
                (self.matched_int_query - self.matched_int_library) ** 2
            ).sum() + (self.unmatched_int_library**2).sum()
            # Guard against divide by zero for identical spectra.
            if not math.isclose(denominator, 0.0):
                return np.log(1 / denominator)
            else:
                return 10.0
        else:
            return 0.0

    def entropy(self, weighted: bool = False) -> float:
        """
        Get the spectral entropy.

        For the original description, see:
        Li, Y. et al. Spectral entropy outperforms MS/MS dot product similarity
        for small-molecule compound identification. Nature Methods 18,
        1524–1531 (2021).

        Parameters
        ----------
        weighted : bool
            Whether to use the unweighted or weighted version of entropy.

        Returns
        -------
        float
            The spectral entropy similarity between the two spectra.
        """
        if self._top is not None:
            raise NotImplementedError(
                "The spectral entropy is not defined when filtering by the "
                "top intensity library peaks"
            )
        elif self.matched_int_query is not None:
            # Entropy of the individual spectra.
            query_entropy = _spectrum_entropy(self.int_query, weighted)
            library_entropy = _spectrum_entropy(self.int_library, weighted)
            # Entropy of the merged spectrum.
            int_merged = (
                np.hstack(
                    [
                        # Element-wise summed intensities of the matched peaks.
                        self.matched_int_query + self.matched_int_library,
                        # Intensities of the unmatched query and library peaks.
                        self.unmatched_int_query,
                        self.unmatched_int_library,
                    ]
                )
                / 2
            )
            merged_entropy = _spectrum_entropy(int_merged, weighted)

            return 1 - (
                2 * merged_entropy - query_entropy - library_entropy
            ) / np.log(4)
        else:
            return 0.0


def _spectrum_entropy(
    spectrum_intensity: np.ndarray, weighted: bool = False
) -> float:
    """
    Compute the entropy of a spectrum from its peak intensities.

    Parameters
    ----------
    spectrum_intensity : np.ndarray
        The intensities of the spectrum peaks.
    weighted : bool
        Whether to use the unweighted or weighted version of entropy.

    Returns
    -------
    float
        The entropy of the given spectrum.
    """
    weight_start, entropy_cutoff = 0.25, 3
    weight_slope = (1 - weight_start) / entropy_cutoff
    spec_entropy = scipy.stats.entropy(spectrum_intensity)
    if not weighted or spec_entropy > entropy_cutoff:
        return spec_entropy
    else:
        weight = weight_start + weight_slope * spec_entropy
        weighted_intensity = spectrum_intensity**weight
        weighted_intensity /= weighted_intensity.sum()
        return scipy.stats.entropy(weighted_intensity)
