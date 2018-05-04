#include <algorithm>
#include <math.h>
#include <tuple>

#include "SpectrumMatch.h"
using namespace ann_solo;

SpectrumSpectrumMatch* SpectrumMatcher::dot(
    Spectrum *query, std::vector<Spectrum*> candidates, double fragment_mz_tolerance, bool allow_shift)
{
    // compute a dot product score between the query spectrum and each candidate spectrum
    SpectrumSpectrumMatch *best_match = NULL;
    for(unsigned int candidate_index = 0; candidate_index < candidates.size(); candidate_index++)
    {
        Spectrum *candidate = candidates[candidate_index];

        // candidate peak indices depend on whether we allow shifts (check all shifted peaks as well) or not
        double precursor_mass_diff = (query->getPrecursorMz() - candidate->getPrecursorMz()) * candidate->getPrecursorCharge();
        // only take peak shifts into account if the mass difference is relevant
        unsigned int num_shifts = allow_shift && fabs(precursor_mass_diff) >= fragment_mz_tolerance ? candidate->getPrecursorCharge() + 1 : 1;
        unsigned int candidate_peak_index[num_shifts];
        for(unsigned int charge = 0; charge < num_shifts; charge++)
        {
            candidate_peak_index[charge] = 0;
        }
        double mass_diff[num_shifts];
        mass_diff[0] = 0;
        for(unsigned int charge = 1; charge < num_shifts; charge++)
        {
            mass_diff[charge] = precursor_mass_diff / charge;
        }

        // find the matching peaks between the query spectrum and the candidate spectrum
        std::vector<std::tuple<float, unsigned int, unsigned int>> peak_matches;
        for(unsigned int query_peak_index = 0; query_peak_index < query->getNumPeaks(); query_peak_index++)
        {
            float query_peak_mass = query->getPeakMass(query_peak_index);
            // advance while there is an excessive mass difference
            for(unsigned int cpi = 0; cpi < num_shifts; cpi++)
            {
                while(candidate_peak_index[cpi] < candidate->getNumPeaks() - 1 &&
                      query_peak_mass - fragment_mz_tolerance > candidate->getPeakMass(candidate_peak_index[cpi]) + mass_diff[cpi])
                {
                    candidate_peak_index[cpi]++;
                }
            }

            // match the peaks within the fragment mass window if possible
            for(unsigned int cpi = 0; cpi < num_shifts; cpi++)
            {
                for(unsigned int index = 0;
                    candidate_peak_index[cpi] + index < candidate->getNumPeaks() &&
                        fabs(query_peak_mass - (candidate->getPeakMass(candidate_peak_index[cpi] + index) + mass_diff[cpi]))
                            <= fragment_mz_tolerance;
                    index++)
                {
                    // slightly penalize matching peaks without an annotation
                    double match_multiplier = 0.0;
                    // unshifted peaks are matched directly
                    if(cpi == 0)
                    {
                        match_multiplier = 1.0;
                    }
                    // shifted peaks with a known charge (and therefore annotation)
                    // should be shifted with a mass difference according to this charge
                    else if(candidate->getPeakCharge(candidate_peak_index[cpi] + index) == cpi)
                    {
                        match_multiplier = 1.0;
                    }
                    // shifted peaks without a known charge
                    // can be shifted with a mass difference according to any charge up to the precursor charge
                    else if(candidate->getPeakCharge(candidate_peak_index[cpi] + index) == 0)
                    {
                        match_multiplier = 2.0 / 3.0;
                    }

                    if(match_multiplier > 0.0)
                    {
                        float query_peak_intensity = query->getPeakIntensity(query_peak_index);
                        float candidate_peak_intensity = candidate->getPeakIntensity(candidate_peak_index[cpi] + index);
                        peak_matches.push_back(std::make_tuple(match_multiplier * query_peak_intensity * candidate_peak_intensity,
                                                               query_peak_index, candidate_peak_index[cpi] + index));
                    }
                }
            }
        }

        // use the most prominent peak matches to compute the score (sort in descending order)
        std::sort(peak_matches.begin(), peak_matches.end(), [](auto &peak_match1, auto &peak_match2) {
                  return std::get<0>(peak_match1) > std::get<0>(peak_match2); });
        SpectrumSpectrumMatch *this_match = new SpectrumSpectrumMatch(candidate_index);
        std::vector<bool> query_peaks_used(query->getNumPeaks(), false);
        std::vector<bool> candidate_peaks_used(candidate->getNumPeaks(), false);
        for(unsigned int peak_match_index = 0; peak_match_index < peak_matches.size(); peak_match_index++)
        {
            std::tuple<double, unsigned int, unsigned int> peak_match = peak_matches[peak_match_index];
            unsigned int query_peak_index = std::get<1>(peak_match);
            unsigned int candidate_peak_index = std::get<2>(peak_match);
            if(query_peaks_used[query_peak_index] == false && candidate_peaks_used[candidate_peak_index] == false)
            {
                this_match->setScore(this_match->getScore() + std::get<0>(peak_match));
                // save the matched peaks
                this_match->addPeakMatch(query_peak_index, candidate_peak_index);
                // make sure these peaks are not used anymore
                query_peaks_used[query_peak_index] = true;
                candidate_peaks_used[candidate_peak_index] = true;
            }
        }

        query_peaks_used.clear();
        candidate_peaks_used.clear();
        peak_matches.clear();

        // retain the match with the highest score
        if(best_match == NULL || best_match->getScore() < this_match->getScore())
        {
            if(best_match != NULL)
            {
                delete best_match;
            }
            best_match = this_match;
        }
        else
        {
            delete this_match;
        }
    }

    return best_match;
}
