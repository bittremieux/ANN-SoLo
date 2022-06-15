import numpy as np


def fidelity_distance(p, q) -> float:
    r"""
    Reference: https://pdodds.w3.uvm.edu/research/papers/others/everything/cha2007a.pdf
    Fidelity distance:

    .. math::

        1-\sum\sqrt{P_{i}Q_{i}}
    """
    return 1 - np.sum(np.sqrt(p * q))


def matusita_distance(p, q) -> float:
    r"""
    Reference: https://pdodds.w3.uvm.edu/research/papers/others/everything/cha2007a.pdf
    Matusita distance:

    .. math::

        \sqrt{\sum(\sqrt{P_{i}}-\sqrt{Q_{i}})^2}
    """
    return np.sqrt(np.sum(np.power(np.sqrt(p) - np.sqrt(q), 2)))


def squared_chord_distance(p, q) -> float:
    r"""
    Reference: https://pdodds.w3.uvm.edu/research/papers/others/everything/cha2007a.pdf
    Squared-chord distance:

    .. math::

        \sum(\sqrt{P_{i}}-\sqrt{Q_{i}})^2
    """
    return np.sum(np.power(np.sqrt(p) - np.sqrt(q), 2))


def bhattacharya_1_distance(p, q) -> float:
    r"""
    Refrence: https://trepo.tuni.fi/bitstream/handle/10024/124353/Distance%20measures%20for%20classi%ef%ac%81cation%20of%20numerical%20features.pdf?sequence=2&isAllowed=y
    Bhattacharya 1 distance:

    .. math::

        (\arccos{(\sum\sqrt{P_{i}Q_{i}})})^2
    """
    s = np.sum(np.sqrt(p * q))
    if s > 1:
        s = 1
    return np.power(np.arccos(s), 2)


def bhattacharya_2_distance(p, q) -> float:
    r"""
    Reference: https://pdodds.w3.uvm.edu/research/papers/others/everything/cha2007a.pdf
    Bhattacharya 2 distance:

    .. math::

        -\ln{(\sum\sqrt{P_{i}Q_{i}})}
    """
    s = np.sum(np.sqrt(p * q))
    if s == 0:
        return np.inf
    else:
        return -np.log(s)


def harmonic_mean_distance(p, q) -> float:
    r"""
    Refrence: https://trepo.tuni.fi/bitstream/handle/10024/124353/Distance%20measures%20for%20classi%ef%ac%81cation%20of%20numerical%20features.pdf?sequence=2&isAllowed=y

	Harmonic mean distance:

    .. math::

        1-2\sum(\frac{P_{i}Q_{i}}{P_{i}+Q_{i}})
    """
    return 1 - 2 * np.sum(p * q / (p + q))


def probabilistic_symmetric_chi_squared_distance(p, q) -> float:
    r"""
    Refrence: https://trepo.tuni.fi/bitstream/handle/10024/124353/Distance%20measures%20for%20classi%ef%ac%81cation%20of%20numerical%20features.pdf?sequence=2&isAllowed=y
    Probabilistic symmetric χ2 distance:

    .. math::

        \frac{1}{2} \times \sum\frac{(P_{i}-Q_{i}\ )^2}{P_{i}+Q_{i}\ }
    """
    return 1 / 2 * np.sum(np.power(p - q, 2) / (p + q))


def ruzicka_distance(p, q) -> float:
    r"""
    Reference: https://pdodds.w3.uvm.edu/research/papers/others/everything/cha2007a.pdf
    Ruzicka distance:

    .. math::

        \frac{\sum{|P_{i}-Q_{i}|}}{\sum{\max(P_{i},Q_{i})}}
    """
    dist = np.sum(np.abs(p - q)) / np.sum(np.maximum(p, q))
    return dist


def roberts_distance(p, q) -> float:
    r"""
    Refrence: https://trepo.tuni.fi/bitstream/handle/10024/124353/Distance%20measures%20for%20classi%ef%ac%81cation%20of%20numerical%20features.pdf?sequence=2&isAllowed=y
    Roberts distance:

    .. math::

        1-\sum\frac{(P_{i}+Q_{i})\frac{\min{(P_{i},Q_{i})}}{\max{(P_{i},Q_{i})}}}{\sum(P_{i}+Q_{i})}
    """
    return 1 - np.sum((p + q) / np.sum(p + q) *
                      np.minimum(p, q) / np.maximum(p, q))


def intersection_distance(p, q) -> float:
    r"""
    Reference: https://pdodds.w3.uvm.edu/research/papers/others/everything/cha2007a.pdf
    Intersection distance:

    .. math::

        1-\frac{\sum\min{(P_{i},Q_{i})}}{\min(\sum{P_{i},\sum{Q_{i})}}}
    """
    return 1 - np.sum(np.minimum(p, q)) / min(np.sum(p), np.sum(q))


def motyka_distance(p, q) -> float:
    r"""
    Reference: https://pdodds.w3.uvm.edu/research/papers/others/everything/cha2007a.pdf
    Motyka distance:

    .. math::

        -\frac{\sum\min{(P_{i},Q_{i})}}{\sum(P_{i}+Q_{i})}
    """
    dist = np.sum(np.minimum(p, q)) / np.sum(p + q)
    return -dist





def baroni_urbani_buser_distance(p, q) -> float:
    r"""
    Refrence: https://trepo.tuni.fi/bitstream/handle/10024/124353/Distance%20measures%20for%20classi%ef%ac%81cation%20of%20numerical%20features.pdf?sequence=2&isAllowed=y
    Baroni-Urbani-Buser distance:

    .. math::

        1-\frac{\sum\min{(P_i,Q_i)}+\sqrt{\sum\min{(P_i,Q_i)}\sum(\max{(P)}-\max{(P_i,Q_i)})}}{\sum{\max{(P_i,Q_i)}+\sqrt{\sum{\min{(P_i,Q_i)}\sum(\max{(P)}-\max{(P_i,Q_i)})}}}}
    """
    if np.max(p) < np.max(q):
        p, q = q, p
    d1 = np.sqrt(np.sum(np.minimum(p, q) * np.sum(max(p) - np.maximum(p, q))))
    return 1 - (np.sum(np.minimum(p, q)) + d1) / \
        (np.sum(np.maximum(p, q)) + d1)


def penrose_size_distance(p, q) -> float:
    r"""
    Refrence: https://trepo.tuni.fi/bitstream/handle/10024/124353/Distance%20measures%20for%20classi%ef%ac%81cation%20of%20numerical%20features.pdf?sequence=2&isAllowed=y
    Penrose size distance:

    .. math::

        \sqrt N\sum{|P_i-Q_i|}
    """
    n = np.sum(p > 0)
    return np.sqrt(n) * np.sum(np.abs(p - q))


def mean_character_distance(p, q) -> float:
    r"""
    Refrence: https://trepo.tuni.fi/bitstream/handle/10024/124353/Distance%20measures%20for%20classi%ef%ac%81cation%20of%20numerical%20features.pdf?sequence=2&isAllowed=y
    Mean character distance:

    .. math::

        \frac{1}{N}\sum{|P_i-Q_i|}
    """
    n = np.sum(p > 0)
    return 1 / n * np.sum(np.abs(p - q))


def lorentzian_distance(p, q) -> float:
    r"""
    Reference: https://pdodds.w3.uvm.edu/research/papers/others/everything/cha2007a.pdf
    Lorentzian distance:

    .. math::

        \sum{\ln(1+|P_i-Q_i|)}
    """
    return np.sum(np.log(1 + np.abs(p - q)))


def penrose_shape_distance(p, q) -> float:
    r"""
    Refrence: https://trepo.tuni.fi/bitstream/handle/10024/124353/Distance%20measures%20for%20classi%ef%ac%81cation%20of%20numerical%20features.pdf?sequence=2&isAllowed=y
    Penrose shape distance:

    .. math::

        \sqrt{\sum((P_i-\bar{P})-(Q_i-\bar{Q}))^2}
    """
    p_avg = np.mean(p)
    q_avg = np.mean(q)
    return np.sqrt(np.sum(np.power((p - p_avg) - (q - q_avg), 2)))


def clark_distance(p, q) -> float:
    r"""
    Reference: https://pdodds.w3.uvm.edu/research/papers/others/everything/cha2007a.pdf
    Clark distance:

    .. math::

        (\frac{1}{N}\sum(\frac{P_i-Q_i}{|P_i|+|Q_i|})^2)^\frac{1}{2}
    """
    n = np.sum(p > 0)
    return np.sqrt(
        1 / n * np.sum(np.power((p - q) / (np.abs(p) + np.abs(q)), 2)))


def hellinger_distance(p, q) -> float:
    r"""
    Reference: https://pdodds.w3.uvm.edu/research/papers/others/everything/cha2007a.pdf
    Hellinger distance:

    .. math::

        \sqrt{2\sum(\sqrt{\frac{P_i}{\bar{P}}}-\sqrt{\frac{Q_i}{\bar{Q}}})^2}
    """
    p_avg = np.mean(p)
    q_avg = np.mean(q)
    return np.sqrt(
        2 *
        np.sum(
            np.power(
                np.sqrt(
                    p /
                    p_avg) -
                np.sqrt(
                    q /
                    q_avg),
                2)))


def whittaker_index_of_association_distance(p, q) -> float:
    r"""
    Refrence: https://trepo.tuni.fi/bitstream/handle/10024/124353/Distance%20measures%20for%20classi%ef%ac%81cation%20of%20numerical%20features.pdf?sequence=2&isAllowed=y
    Whittaker index of association distance:

    .. math::

        \frac{1}{2}\sum|\frac{P_i}{\bar{P}}-\frac{Q_i}{\bar{Q}}|
    """
    p_avg = np.mean(p)
    q_avg = np.mean(q)
    return 1 / 2 * np.sum(np.abs(p / p_avg - q / q_avg))


def symmetric_chi_squared_distance(p, q) -> float:
    r"""
    Refrence: https://trepo.tuni.fi/bitstream/handle/10024/124353/Distance%20measures%20for%20classi%ef%ac%81cation%20of%20numerical%20features.pdf?sequence=2&isAllowed=y
    Symmetric χ2 distance:

    .. math::

        \sqrt{\sum{\frac{\bar{P}+\bar{Q}}{N(\bar{P}+\bar{Q})^2}\frac{(P_i\bar{Q}-Q_i\bar{P})^2}{P_i+Q_i}\ }}
    """
    p_avg = np.mean(p)
    q_avg = np.mean(q)
    n = np.sum(p > 0)

    d1 = (p_avg + q_avg) / (n * np.power(p_avg + q_avg, 2))
    return np.sqrt(d1 * np.sum(np.power(p * q_avg - q * p_avg, 2) / (p + q)))


def improved_similarity_distance(p, q) -> float:
    r"""
    Refrence: https://trepo.tuni.fi/bitstream/handle/10024/124353/Distance%20measures%20for%20classi%ef%ac%81cation%20of%20numerical%20features.pdf?sequence=2&isAllowed=y
    Improved Similarity Index:

    .. math::

        \sqrt{\frac{1}{N}\sum\{\frac{P_i-Q_i}{P_i+Q_i}\}^2}
    """
    n = np.sum(p > 0)
    return np.sqrt(1 / n * np.sum(np.power((p - q) / (p + q), 2)))


def absolute_value_distance(p, q) -> float:
    r"""
    Refrence: https://trepo.tuni.fi/bitstream/handle/10024/124353/Distance%20measures%20for%20classi%ef%ac%81cation%20of%20numerical%20features.pdf?sequence=2&isAllowed=y
    Absolute Value Distance:

    .. math::

        \frac { \sum(|Q_i-P_i|)}{\sum P_i}

    """
    dist = np.sum(np.abs(q - p)) / np.sum(p)
    return dist


def spectral_contrast_angle_distance(p, q) -> float:
    r"""
    Refrence: https://trepo.tuni.fi/bitstream/handle/10024/124353/Distance%20measures%20for%20classi%ef%ac%81cation%20of%20numerical%20features.pdf?sequence=2&isAllowed=y
    Spectral Contrast Angle distance.
    Please note that the value calculated here is :math:`\cos\theta`.
    If you want to get the :math:`\theta`, you can calculate with: :math:`\arccos(1-distance)`

    .. math::

        1 - \frac{\sum{Q_iP_i}}{\sqrt{\sum Q_i^2\sum P_i^2}}
    """
    return 1 - np.sum(q * p) / \
        np.sqrt(np.sum(np.power(q, 2)) * np.sum(np.power(p, 2)))


def wave_hedges_distance(p, q) -> float:
    r"""
    Reference: https://pdodds.w3.uvm.edu/research/papers/others/everything/cha2007a.pdf
    Wave Hedges distance:

    .. math::

        \sum\frac{|P_i-Q_i|}{\max{(P_i,Q_i)}}
    """
    return np.sum(np.abs(p - q) / np.maximum(p, q))





def divergence_distance(p, q) -> float:
    r"""
    Reference: https://pdodds.w3.uvm.edu/research/papers/others/everything/cha2007a.pdf
    Divergence distance:

    .. math::

        2\sum\frac{(P_i-Q_i)^2}{(P_i+Q_i)^2}
    """
    return 2 * np.sum((np.power(p - q, 2)) / np.power(p + q, 2))


def vicis_symmetric_chi_squared_3_distance(p, q) -> float:
    r"""
    Reference: https://pdodds.w3.uvm.edu/research/papers/others/everything/cha2007a.pdf
    Vicis-Symmetric χ2 3 distance:

    .. math::

        \sum\frac{(P_i-Q_i)^2}{\max{(P_i,Q_i)}}
    """
    return np.sum(np.power(p - q, 2) / np.maximum(p, q))
