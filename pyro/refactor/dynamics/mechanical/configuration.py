import dataclasses


@dataclasses.dataclass
class MechanicalConfiguration:
    dof: int
    mass: float
    inertia: float
