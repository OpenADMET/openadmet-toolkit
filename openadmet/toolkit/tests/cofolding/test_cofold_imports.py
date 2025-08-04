# import the classes from the module
# we don't test on GPUs yet so the best we can do is to check if the classes are importable
def test_cofold_imports():
    from openadmet.toolkit.cofolding.cofold_base import CoFoldingEngine # noqa: F401 F403
    from openadmet.toolkit.cofolding.chai1 import Chai1CoFoldingEngine # noqa: F401 F403
    from openadmet.toolkit.cofolding.boltz import BoltzCoFoldingEngine # noqa: F401 F403
