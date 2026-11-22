import re

import numpy as np


def ac_metrics(freq, response):
    """Return DC gain (dB), unity-gain frequency (Hz), and phase margin (deg)."""
    magnitude_db = 20 * np.log10(np.abs(response))
    phase = np.degrees(np.unwrap(np.angle(response)))
    log_frequency = np.log10(freq)
    dc_gain = magnitude_db[0]

    below_unity = np.where(magnitude_db < 0)[0]
    if len(below_unity) == 0 or below_unity[0] == 0:
        return dc_gain, np.nan, np.nan
    index = below_unity[0]
    crossing_log_frequency = np.interp(
        0.0,
        [magnitude_db[index], magnitude_db[index - 1]],
        [log_frequency[index], log_frequency[index - 1]],
    )
    crossing_phase = np.interp(
        crossing_log_frequency,
        [log_frequency[index - 1], log_frequency[index]],
        [phase[index - 1], phase[index]],
    )
    return (
        dc_gain,
        10**crossing_log_frequency,
        180 + (crossing_phase - phase[0]),
    )


def parse_op(stdout):
    """Collect every scalar 'name = value' line emitted by ngspice."""
    return {
        match.group(1).lower(): float(match.group(2))
        for match in re.finditer(r"^(\S+) = (\S+)$", stdout, re.MULTILINE)
    }
