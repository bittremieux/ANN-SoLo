from typing import Optional

import numpy as np
import scipy.spatial.distance
import scipy.special
import scipy.stats

from ann_solo import spectrum


class SpectrumSimilarityFactory:
    def __init__(
        self, ssm: spectrum.SpectrumSpectrumMatch, top: Optional[int] = None
    ):
        """
        Instantiate the `SpectrumSimilarityFactory` to compute various spectrum
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
        self.peak_matches = ssm.peak_matches
        if len(self.peak_matches) > 0:
            self.matched_mz_query = self.mz_query[self.peak_matches[:, 0]]
            self.matched_int_query = self.int_query[self.peak_matches[:, 0]]
            self.matched_mz_library = self.mz_library[self.peak_matches[:, 1]]
            self.matched_int_library = self.int_library[self.peak_matches[:, 1]]
            # Filter the peak matches by the `top` highest intensity peaks in
            # the library spectrum.
            if top is not None:
                library_top_i = np.argpartition(self.int_library, -top)[-top:]
                self.top_i_mask = np.isin(
                    self.peak_matches[:, 1], library_top_i, assume_unique=True
                )
        else:
            self.matched_mz_query, self.matched_int_query = None, None
            self.matched_mz_library, self.matched_int_library = None, None



    def cosine(self, topOnly: bool = False) -> float:
        """
        Get the cosine similarity.

        For the original description, see:
        Bittremieux, W., Meysman, P., Noble, W. S. & Laukens, K. Fast open
        modification spectral library searching through approximate nearest
        neighbor indexing. Journal of Proteome Research 17, 3463–3474 (2018).

        Parameters
        ----------
        topOnly: bool = False
            Compute similarity measure for top peaks only.

        Returns
        -------
        float
            The cosine similarity between the two spectra.
        """
        if self.matched_int_query is not None \
                and self.matched_int_library is not None:
            if topOnly and sum(self.top_i_mask)<2:
                return 0.0
            elif topOnly:
                return np.dot(self.matched_int_query[self.top_i_mask],
                              self.matched_int_library[self.top_i_mask])
            else:
                return np.dot(self.matched_int_query,
                              self.matched_int_library)
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
        return len(self.matched_mz_query) if self.matched_mz_query is not None else 0

    def frac_n_peaks_query(self) -> float:
        """
        Get the number of shared peaks as a fraction of the number of peaks in
        the query spectrum.

        Returns
        -------
        float
            The fraction of shared peaks in the query spectrum.
        """
        if self.matched_mz_query is not None:
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
        if self.matched_mz_library is not None:
            return len(self.matched_mz_library) / len(self.mz_library)
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
        if self.matched_int_query is not None:
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
            return self.matched_int_library.sum() / self.int_library.sum()
        else:
            return 0.0

    def mean_squared_error(self, axis: str, topOnly: bool = False) -> float:
        """
        Get the mean squared error (MSE) of peak matches.

        Parameters
        ----------
        axis : str
            Calculate the MSE between the m/z values ("mz") or intensity values
            ("intensity") of the matched peaks.
        topOnly: bool = False
            Compute similarity measure for top peaks only.

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
        if topOnly and sum(self.top_i_mask)<2:
            return 0.0
        elif axis == "mz" and topOnly:
            arr1, arr2 = self.matched_mz_query[self.top_i_mask],\
                         self.matched_mz_library[self.top_i_mask]
        elif axis == "intensity" and topOnly:
            arr1, arr2 = self.matched_int_query[self.top_i_mask],\
                         self.matched_int_library[self.top_i_mask]
        elif axis == "mz" and not topOnly:
            arr1, arr2 = self.matched_int_query, self.matched_int_library
        elif axis == "intensity" and not topOnly:
            arr1, arr2 = self.matched_int_query, self.matched_int_library
        else:
            raise ValueError("Unknown axis specified")
        if arr1 is not None and arr2 is not None:
            return ((arr1 - arr2) ** 2).sum() / len(self.mz_query)
        else:
            return np.inf

    def spectral_contrast_angle(self, topOnly: bool = False) -> float:
        """
        Get the spectral contrast angle.

        For the original description, see:
        Toprak, U. H. et al. Conserved peptide fragmentation as a benchmarking
        tool for mass spectrometers and a discriminating feature for targeted
        proteomics. Molecular & Cellular Proteomics 13, 2056–2071 (2014).

        Parameters
        ----------
        topOnly: bool = False
            Compute similarity measure for top peaks only.

        Returns
        -------
        float
            The spectral contrast angle between the two spectra.
        """
        return 1 - 2 * np.arccos(self.cosine(topOnly)) / np.pi


    def hypergeometric_score(self,min_mz: int, max_mz: int, bin_size: float) \
                            -> float:
        """
        Get the hypergeometric score of peak matches between two spectra.

        The hypergeometric score measures the probability of obtaining more than
        the observed number of peak matches by random chance, which follows a
        hypergeometric distribution.

        For the original description, see:
        Dasari, S. et al. Pepitome: Evaluating improved spectral library search for
        identification complementarity and quality assessment. Journal of Proteome
        Research 11, 1686–1695 (2012).

        Parameters
        ----------
        min_mz : int
            The minimum mz provided in the config file.
        max_mz : int
            The maximum mz provided in the config file.
        bin_size : int
            The bin size provided in the config file.

        Returns
        -------
        float
            The hypergeometric score of peak matches.
        """
        n_library_peaks = len(self.mz_library)
        n_matched_peaks = len(self.matched_mz_library)
        n_peak_bins, _, _ = spectrum.get_dim(
            min_mz, max_mz, bin_size
        )
        return sum(
            [
                (
                    scipy.special.comb(n_library_peaks, i)
                    * scipy.special.comb(
                        n_peak_bins - n_library_peaks, n_library_peaks - i
                    )
                )
                / scipy.special.comb(n_peak_bins, n_library_peaks)
                for i in range(n_matched_peaks + 1, n_library_peaks)
            ]
        )


    def kendalltau(self) -> float:
        """
        Get the Kendall-Tau score of peak matches between two spectra.

        The Kendall-Tau score measures the correspondence between the intensity
        ranks of the set of peaks matched between spectra.

        For the original description, see:
        Dasari, S. et al. Pepitome: Evaluating improved spectral library search for
        identification complementarity and quality assessment. Journal of Proteome
        Research 11, 1686–1695 (2012).

        Returns
        -------
        float
            The hypergeometric score of peak matches.
        """
        return -1 if not len(self.matched_int_query) else \
            scipy.stats.kendalltau(
            self.matched_int_query,
            self.matched_int_library,
        )[0]


    def ms_for_id_v1(self) -> float:
        """
        Compute the MSforID (v1) similarity between two spectra.

        For the original description, see:
        Pavlic, M., Libiseller, K. & Oberacher, H. Combined use of ESI–QqTOF-MS and
        ESI–QqTOF-MS/MS with mass-spectral library search for qualitative analysis
        of drugs. Analytical and Bioanalytical Chemistry 386, 69–82 (2006).

        Returns
        -------
        float
            The MSforID (v1) similarity between both spectra.
        """
        return 0 if not len(self.matched_int_query) else \
            len(self.matched_int_query) ** 4 / (
                len(self.mz_query)
                * len(self.mz_library)
                * max(
                    np.abs(
                        self.matched_int_query
                        - self.matched_int_library
                    ).sum(),
                    np.finfo(float).eps,
                )
                ** 0.25
            )


    def ms_for_id_v2(self) -> float:
        """
        Compute the MSforID (v2) similarity between two spectra.

        For the original description, see:
        Oberacher, H. et al. On the inter-instrument and the inter-laboratory
        transferability of a tandem mass spectral reference library: 2.
        Optimization and characterization of the search algorithm: About an
        advanced search algorithm for tandem mass spectral reference libraries.
        Journal of Mass Spectrometry 44, 494–502 (2009).

        Returns
        -------
        float
            The MSforID (v2) similarity between both spectra.
        """
        return 0 if not len(self.matched_int_query) else \
            (len(self.matched_int_query) ** 4
            * (
                self.int_query.sum()
                + 2 * self.int_library.sum()
            )
            ** 1.25
        ) / (
            (len(self.mz_query) + 2
             * len(self.mz_library)) ** 2
            + np.abs(
                self.matched_int_query
                - self.matched_int_library
            ).sum()
            + np.abs(
                self.matched_mz_query
                - self.matched_mz_library
            ).sum()
        )


    def manhattan(self) -> float:
        """
        Get the Manhattan distance between two spectra.

        Returns
        -------
        float
            The Manhattan distance between both spectra.
        """
        # Matching peaks.
        dist = np.abs(
            self.matched_int_query
            - self.matched_int_library
        ).sum()
        # Unmatched peaks in the query spectrum.
        dist += self.int_query[
            np.setdiff1d(
                np.arange(len(self.int_query)),
                self.peak_matches[:, 0],
                assume_unique=True,
            )
        ].sum()
        # Unmatched peaks in the library spectrum.
        dist += self.int_library[
            np.setdiff1d(
                np.arange(len(self.int_library)),
                self.peak_matches[:, 1],
                assume_unique=True,
            )
        ].sum()
        return dist

    def euclidean(self) -> float:
        """
        Get the Euclidean distance between two spectra.

        Returns
        -------
        float
            The Euclidean distance between both spectra.
        """
        # Matching peaks.
        dist = (
                (
                    self.matched_int_query
                    - self.matched_int_library
                )
                ** 2
        ).sum()
        # Unmatched peaks in the query spectrum.
        dist += (
                self.int_query[
                    np.setdiff1d(
                        np.arange(len(self.int_query)),
                        self.peak_matches[:, 0],
                        assume_unique=True,
                    )
                ]
                ** 2
        ).sum()
        # Unmatched peaks in the library spectrum.
        dist += (
                self.int_library[
                    np.setdiff1d(
                        np.arange(len(self.int_library)),
                        self.peak_matches[:, 1],
                        assume_unique=True,
                    )
                ]
                ** 2
        ).sum()
        return np.sqrt(dist)

    def chebyshev(self) -> float:
        """
        Get the Chebyshev distance between two spectra.

        Returns
        -------
        float
            The Chebyshev distance between both spectra.
        """
        # Matching peaks.
        dist = np.abs(
            self.matched_int_query
            - self.matched_int_library
        )
        # Unmatched peaks in the query spectrum.
        dist = np.hstack(
            (
                dist,
                self.int_query[
                    np.setdiff1d(
                        np.arange(len(self.int_query)),
                        self.peak_matches[:, 0],
                        assume_unique=True,
                    )
                ],
            )
        )
        # Unmatched peaks in the library spectrum.
        dist = np.hstack(
            (
                dist,
                self.int_library[
                    np.setdiff1d(
                        np.arange(len(self.int_library)),
                        self.peak_matches[:, 1],
                        assume_unique=True,
                    )
                ],
            )
        )
        return dist.max()

    def pearsonr(self, topOnly: bool = False) -> float:
        """
        Get the Pearson correlation between peak matches in two spectra.

        Parameters
        ----------
        topOnly: bool = False
            Compute similarity measure for top peaks only.

        Returns
        -------
        float
            The Pearson correlation of peak matches.
        """
        # FIXME: Use all library spectrum peaks.
        peaks_query, peaks_library = self.matched_int_query, \
                                     self.matched_int_library
        if topOnly and sum(self.top_i_mask)<2:
            return 0.0
        elif topOnly:
            peaks_query = self.matched_int_query[self.top_i_mask]
            peaks_library = self.matched_int_library[self.top_i_mask]

        return 0.0 if len(peaks_query) > 1 else \
                scipy.stats.pearsonr(peaks_query, peaks_library)[0]

    def spearmanr(self, topOnly: bool = False) -> float:
        """
        Get the Spearman correlation between peak matches in two spectra.

        Parameters
        ----------
        topOnly: bool = False
            Compute similarity measure for top peaks only.

        Returns
        -------
        float
            The Spearman correlation of peak matches.
        """
        # FIXME: Use all library spectrum peaks.
        peaks_query, peaks_library = self.matched_int_query, \
                                     self.matched_int_library
        if topOnly and sum(self.top_i_mask)<2:
            return 0.0
        elif topOnly:
            peaks_query = self.matched_int_query[self.top_i_mask]
            peaks_library = self.matched_int_library[self.top_i_mask]

        return 0.0 if len(peaks_query) > 1 else \
            scipy.stats.spearmanr(peaks_query, peaks_library)[0]

    def braycurtis(self) -> float:
        """
        Get the Bray-Curtis distance between two spectra.

        The Bray-Curtis distance is defined as:

        .. math::
           \\sum{|u_i-v_i|} / \\sum{|u_i+v_i|}

        Returns
        -------
        float
            The Bray-Curtis distance between both spectra.
        """
        numerator = np.abs(
            self.matched_int_query
            - self.matched_int_library
        ).sum()
        denominator = (
                self.matched_int_query
                + self.matched_int_library
        ).sum()
        query_unique = self.int_query[
            np.setdiff1d(
                np.arange(len(self.int_query)),
                self.peak_matches[:, 0],
                assume_unique=True,
            )
        ].sum()
        library_unique = self.int_library[
            np.setdiff1d(
                np.arange(len(self.int_library)),
                self.peak_matches[:, 1],
                assume_unique=True,
            )
        ].sum()
        numerator += query_unique + library_unique
        denominator += query_unique + library_unique
        return numerator / denominator

    def canberra(self) -> float:
        """
        Get the Canberra distance between two spectra.

        Returns
        -------
        float
            The canberra distance between both spectra.
        """
        dist = scipy.spatial.distance.canberra(
            self.matched_int_query,
            self.matched_int_library,
        )
        # Account for unmatched peaks in the query and library spectra.
        dist += len(self.mz_query) - len(self.matched_mz_query)
        dist += len(self.mz_library) - len(self.matched_mz_query)
        return dist

    def ruzicka(self) -> float:
        """
        Compute the Ruzicka similarity between two spectra.

        Returns
        -------
        float
            The Ruzicka similarity between both spectra.
        """
        numerator = np.minimum(
            self.matched_int_query,
            self.matched_int_library,
        ).sum()
        denominator = np.maximum(
            self.matched_int_query,
            self.matched_int_library,
        ).sum()
        # Account for unmatched peaks in the query and library spectra.
        denominator += self.int_query[
            np.setdiff1d(
                np.arange(len(self.int_query)),
                self.peak_matches[:, 0],
                assume_unique=True,
            )
        ].sum()
        denominator += self.int_library[
            np.setdiff1d(
                np.arange(len(self.int_library)),
                self.peak_matches[:, 1],
                assume_unique=True,
            )
        ].sum()
        return numerator / denominator

    def scribe_fragment_acc(self, topOnly: bool = False) -> float:
        """
        Get the Scribe fragmentation accuracy between two spectra.

        For the original description, see:
        Searle, B. C. et al. Scribe: next-generation library searching for DDA
        experiments. ASMS 2022.

        Parameters
        ----------
        topOnly: bool = False
            Compute similarity measure for top peaks only.

        Returns
        -------
        float
            The Scribe fragmentation accuracy between both spectra.
        """
        # FIXME: Use all library spectrum peaks.
        peaks_query, peaks_library = self.matched_int_query, \
                                     self.matched_int_library
        if topOnly and sum(self.top_i_mask)<2:
            return 0.0
        elif topOnly:
            peaks_query = self.matched_int_query[self.top_i_mask]
            peaks_library = self.matched_int_library[self.top_i_mask]
        return np.log(
            1
            / max(
                0.001,  # Guard against infinity for identical spectra.
                (
                        (
                                peaks_query / peaks_query.sum()
                                - peaks_library / peaks_library.sum()
                        )
                        ** 2
                ).sum(),
            ),
        )

    def entropy(self, weighted: bool = False) -> float:
        """
        Get the entropy between two spectra.

        For the original description, see:
        Li, Y. et al. Spectral entropy outperforms MS/MS dot product similarity for
        small-molecule compound identification. Nature Methods 18, 1524–1531
        (2021).

        Parameters
        ----------
        weighted : bool
            Whether to use the unweighted or weighted version of entropy.

        Returns
        -------
        float
            The entropy between both spectra.
        """
        query_entropy = self._spectrum_entropy(self.int_query,weighted)
        library_entropy = self._spectrum_entropy(self.int_library, weighted)
        merged_entropy = self._spectrum_entropy(self._merge_entropy(),
                                                        weighted)
        return 2 * merged_entropy - query_entropy - library_entropy

    def _spectrum_entropy(
        self,spectrum_intensity: np.ndarray, weighted: bool = False
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


    def _merge_entropy(self) -> np.ndarray:
        """
        Merge two spectra prior to entropy calculation of the spectrum-spectrum
        match.


        Returns
        -------
        np.ndarray
            NumPy array with the intensities of the merged peaks summed.
        """
        # Initialize with the query spectrum peaks.
        merged = self.int_query.copy()
        # Sum the intensities of matched peaks.
        merged[self.peak_matches[:, 0]] += self.int_library[
                                                    self.peak_matches[:, 1]
                                                    ]
        # Append the unmatched library spectrum peaks.
        merged = np.hstack(
            (
                merged,
                self.int_library[
                    np.setdiff1d(
                        np.arange(len(self.int_library)),
                        self.peak_matches[:, 1],
                        assume_unique=True,
                    )
                ],
            )
        )
        return merged
