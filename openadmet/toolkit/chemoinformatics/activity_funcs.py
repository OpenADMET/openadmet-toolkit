import numpy as np
from openff.units import unit
from typing import Optional


def calculate_pac50(activity:float, input_unit_str:str) -> float:
    """A function to calculate pAC50 from an activity measure in units of molarity (M).
    Molarity can be in M, mM, uM, µM, or nM.
    Acceptable activity measures are: EC50, IC50, XC50, AC50.

    Parameters
    ----------
    activity : float
        activity measure in units of molarity
    input_unit_str : str
        string of molarity units, one of "M", "mM", "uM", "µM", or "nM"

    Returns
    -------
    float
        p(activity measure)

    Raises
    ------
    ValueError
        Unsupported molarity unit. Must be one of input_unit_str
    ValueError
        Negative activity measure values are not allowed.
    """
    # First, convert the activity measure to proper molarity units
    unit_map = {
        "M": unit.molar,
        "mM": unit.millimolar,
        "µM": unit.micromolar,
        "uM": unit.micromolar,
        "nM": unit.nanomolar,
    }

    input_unit = unit_map.get(input_unit_str)

    if input_unit is None:
        raise ValueError(f"Unsupported molarity unit: {input_unit_str}. Must be one of {unit_map.keys()}.")

    # Get activity with the appropriate molarity units
    activity_m = (activity * input_unit).to(unit.molar)
    # Now, we can calculate the pAC50 value
    if activity > 0:
        return -np.log10(activity_m.magnitude)
    else:
        raise ValueError("Whoops! Negative activity measure values are not allowed.")

def pIC50_to_Ki(pIC50:float, S: Optional[float] = None, Km: Optional[float] = None) -> float:
    """A function to approximate inhibition constant (Ki) from pIC50 with the Cheng-Prusoff Equation:
        Ki = IC50/(1+ S/Km)
    If S and Km are not provided, then the assumption is that S/Km = 0, aka there is negligible substrate concentration such that Ki ≈ IC50.


    Parameters
    ----------
    pIC50 : float
        log transform of inihibitory concentration (IC)
    S : Optional[float], optional
        substrate concentration, by default None
    Km : Optional[float], optional
        Michaelis-Menten constant, aka when the substrate concentration at which the enzyme is at half os maximum velocity (Vmax); it is a measure of substrate affinity where low Km means higher affinity, by default None

    Returns
    -------
    float
        inhibition constant (Ki) in units of molarity (M)
    """
    if S is None and Km is None:
        return 10 ** (-1 * pIC50) * unit.molar
    else:
        return 10 ** (-1 * pIC50) / (1 + S / Km)

def ki_to_dg(ki:unit.Quantity, input_unit_str:str, temp_rxn:unit.Quantity = 298.15 * unit.kelvin) -> float:
    """A function to calculate Gibbs free energy (delta G, or dg) from p(activity measure).
    Final output is Gibbs free energy in units of kJ/mol

    This calculation assumes that:
    - Ki is the true thermodynamic dissociation constant
    - Ideal solution
    - Constant temperature

    Notes:
        The equation is G = RT ln(Ki) where
            G = Gibbs free energy
            R = gas constant, 8.3145 J/(K*mol)
            T = temperature of reaction in Kelvin
            ln(Ki) = natural log of inhibition constant

    Args:
        ki (float): inhibition or dissociation constant of a small molecule inhibitor to a protein anti-target
        input_unit_str (str): string of molarity units, one of "M", "mM", "uM", "µM", or "nM"
        temp_rxn (float): temperature at which the reaction takes place, default is 25 C or 298.15 K

    Returns:
        float: Gibbs free energy (dg) in units of kJ/mol
    """
    # First, convert the activity measure to proper molarity units
    unit_map = {
        "M": unit.molar,
        "mM": unit.millimolar,
        "µM": unit.micromolar,
        "uM": unit.micromolar,
        "nM": unit.nanomolar,
    }

    input_unit = unit_map.get(input_unit_str)

    if input_unit is None:
        raise ValueError(f"Unsupported molarity unit: {input_unit_str}. Must be one of {unit_map.keys()}.")

    # Get activity with the appropriate molarity units
    ki_m = (ki * input_unit).to(unit.molar)

    if ki_m > 0:
        dg = (
            unit.molar_gas_constant *
            temp_rxn.to(unit.kelvin) *
            np.log(ki_m/unit.molar)
            ).to(unit.kilojoule_per_mole)
        return dg
    else:
        raise ValueError("Ah non! Your inhibition constant is negative. Gibbs free energy cannot be calculated.")
