from enum import Flag, auto
from dataclasses import dataclass


class FacetBCTypes(Flag):
    """ Point types for time dependent problems. """

    Dirich = auto()  # Fixed value point
    Neuman = auto()  # Fixed gradient
    Both = Dirich | Neuman  # Both Dirichlet and Neumann BC enforced on point.
    Farfield = auto()  # Farfield boundary condition
    Inlet = auto()


@dataclass
class Facet:
    """"""
    edge_type: list[FacetBCTypes]
    U: list[float| None] = None
    dUdn: list[float | None] = None
    euler_wall: bool = False
    tag: str = None

    def __post_init__(self):
        if self.U is None:
            self.U = [None] * len(self.edge_type)
        if self.dUdn is None:
            self.dUdn = [None] * len(self.edge_type)

        for e, u, dudn in zip(self.edge_type, self.U, self.dUdn, strict=True):
            if FacetBCTypes.Dirich in e:
                assert u is not None, "Dirichlet BC requires a value."
                assert dudn is None, "Dirichlet BC does not require a gradient."

            if FacetBCTypes.Neuman in e:
                assert dudn is not None, "Neumann BC requires a gradient."
                assert u is None, "Neumann BC does not require a value."

        # Replace Nones in U and dUdn with const
        self.U = [float('NaN') if u is None else u for u in self.U]
        self.dUdn = [float('NaN') if d is None else d for d in self.dUdn]

    def dirichlet(self):
        return [FacetBCTypes.Dirich in e for e in self.edge_type]

    def neumann(self):
        return [FacetBCTypes.Neuman in e for e in self.edge_type]

    def farfield(self) -> bool:
        farfield = [FacetBCTypes.Farfield in e for e in self.edge_type]
        is_farfield = all(farfield)
        assert is_farfield or not any(farfield), f"All boundary must be farfield if any are: {farfield}"
        return is_farfield

    def inlet(self) -> bool:
        inlet = [FacetBCTypes.Inlet in e for e in self.edge_type]
        is_inlet = all(inlet)
        assert is_inlet or not any(inlet), f"All boundary must be inlet if any are: {inlet}"
        return is_inlet
