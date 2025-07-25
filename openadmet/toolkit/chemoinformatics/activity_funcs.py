import numpy as np
from openff.units import unit

def calculate_pac50(activity:float, input_unit:str) -> float:
    """A function to calculate pAC50 from an activity measure in units of molarity (M).
    Molarity can be in M, mM, uM, µM, or nM.
    Acceptable activity measures are: EC50, IC50, XC50, AC50.

    Args:
        activity (float): activity measure in units of molarity
        input_unit_str (str): string of molarity units, one of "M", "mM", "uM", "µM", or "nM"

    Returns:
        float: p(activity measure)
    """

    # First, convert the activity measure to proper molarity units
    unit_map = {
        "M": unit.molar,
        "mM": unit.millimolar,
        "µM": unit.micromolar,
        "uM": unit.micromolar,
        "nM": unit.nanomolar,
    }

    input_molar = unit_map.get(input_unit)

    if input_molar is None:
        raise ValueError(f"Unsupported molarity unit: {input_molar}. Must be one of {unit_map.keys()}.")

    # Get activity with the appropriate molarity units
    activity_m = (activity * input_molar).to(unit.molar)

    # Now, we can calculate the pAC50 value
    if activity > 0:
        return -np.log10(activity_m.magnitude)
    else:
        raise ValueError("Whoops! Negative activity measure values are not allowed.")

def pac50_to_ki(pac50:float) -> float:
    """A function to convert pAC50 value to inhibition constant (Ki).

    Args:
        pac50 (float): p(activity measure)

    Returns:
        float: inhibition constant (Ki) in units of molarity (M)
    """
    return 10 ** (-1 * pac50) * unit.molar


def ki_to_dg(ki:unit.Quantity, temp_rxn:unit.Quantity = 298.15 * unit.kelvin) -> float:
    """A function to calculate Gibbs free energy (delta G, or dg) from p(activity measure).
    Final output is Gibbs free energy in units of kJ/mol

    Notes:
        The equation is G = RT ln(Ki) where
            G = Gibbs free energy
            R = gas constant, 8.3145 J/(K*mol)
            T = temperature of reaction in Kelvin
            ln(Ki) = natural log of inhibition constant

    Args:
        ki (float): inhibition or dissociation constant of a small molecule inhibitor to a protein anti-target
        temp_rxn (float): temperature at which the reaction takes place, default is 25 C or 298.15 K

    Returns:
        float: Gibbs free energy (dg) in units of kJ/mol
    """
    if ki > 0:
        dg = (
            unit.molar_gas_constant *
            temp_rxn.to(unit.kelvin) *
            np.log(ki/unit.molar)
            ).to(unit.kilojoule_per_mole)
        return dg
    else:
        raise ValueError("Ah non! Your inhibition constant is negative. Gibbs free energy cannot be calculated.")
