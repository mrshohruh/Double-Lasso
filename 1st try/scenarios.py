from dataclasses import dataclass

@dataclass
class SimulationScenario:
    name: str
    n: int = 200
    p: int = 100
    s: int = 5
    beta1: float = 2.0
    rho: float = 0.2
    R: int = 500     
    c: float = 1.1
    ci_level: float = 0.95
    seed: int = 123


def get_scenarios() -> list[SimulationScenario]:
    return [
        SimulationScenario(
            name="baseline",
            n=200,
            p=100,
            s=5,
            beta1=2.0,
            rho=0.2,
            R=500,
            c=1.1,
            ci_level=0.95,
            seed=123,
        ),
        SimulationScenario(
            name="n_sweep_small",
            n=120,
            p=100,
            s=5,
            beta1=2.0,
            rho=0.2,
            R=500,
            c=1.1,
            ci_level=0.95,
            seed=123,
        ),
        SimulationScenario(
            name="n_sweep_large",
            n=320,
            p=100,
            s=5,
            beta1=2.0,
            rho=0.2,
            R=500,
            c=1.1,
            ci_level=0.95,
            seed=123,
        ),
    ]
