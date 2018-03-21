#ifndef SPECTRUMMATCH_H
#define SPECTRUMMATCH_H

#include <cstdint>
#include <utility>
#include <vector>

namespace ann_solo
{
    class Spectrum
    {
        public:
            Spectrum(double precursor_mz, unsigned int precursor_charge, unsigned int num_peaks,
                     float *masses, float *intensities, uint8_t *charges) :
                m_precursor_mz(precursor_mz), m_precursor_charge(precursor_charge), m_num_peaks(num_peaks),
                m_masses(masses), m_intensities(intensities), m_charges(charges) {}
            ~Spectrum() {}

            double getPrecursorMz() const { return m_precursor_mz; }
            unsigned int getPrecursorCharge() const { return m_precursor_charge; }
            unsigned int getNumPeaks() const { return m_num_peaks; }
            float getPeakMass(unsigned int peak_index) const { return *(m_masses + peak_index); }
            float getPeakIntensity(unsigned int peak_index) const { return *(m_intensities + peak_index); }
            uint8_t getPeakCharge(unsigned int peak_index) const { return *(m_charges + peak_index); }

        private:
            double m_precursor_mz;
            unsigned int m_precursor_charge;
            unsigned int m_num_peaks;
            float *m_masses;
            float *m_intensities;
            uint8_t *m_charges;
    };

    class SpectrumSpectrumMatch
    {
        public:
            SpectrumSpectrumMatch(unsigned int candidate_index) :
                m_candidate_index(candidate_index), m_score(0), m_peak_matches(new std::vector<std::pair<unsigned int, unsigned int>>()) {}
            ~SpectrumSpectrumMatch() { m_peak_matches->clear(); delete m_peak_matches; }

            unsigned int getCandidateIndex() const { return m_candidate_index; }
            double getScore() const { return m_score; }
            void setScore(double score) { m_score = score; }
            void addPeakMatch(unsigned int peak_index1, unsigned int peak_index2) { m_peak_matches->push_back(std::make_pair(peak_index1, peak_index2)); }
            std::vector<std::pair<unsigned int, unsigned int>>* getPeakMatches() const { return m_peak_matches; }

        private:
            unsigned int m_candidate_index;
            double m_score;
            std::vector<std::pair<unsigned int, unsigned int>> *m_peak_matches;
    };

    class SpectrumMatcher
    {
        public:
            SpectrumMatcher() {}
            ~SpectrumMatcher() {}

            SpectrumSpectrumMatch* dot(Spectrum *query, std::vector<Spectrum*> candidates, double fragment_mz_tolerance, bool allow_shift);
    };
}

#endif
