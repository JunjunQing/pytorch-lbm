"""Boundary conditions for LBM.

Link bounce-back: for fluid nodes adjacent to solid, populations that would
stream to/from solid are reflected in-place. This keeps mass in the fluid domain.

For fluid node x, direction i with x-e_i being solid (incoming from solid):
    f_new[i][x] = f_out[opp(i)][x]   (reflect outgoing population back)
"""
import torch


class BounceBack:
    """Link bounce-back: reflect populations at solid-fluid interface."""

    def __init__(self, lattice, solid_mask: torch.Tensor, device="cuda"):
        self.lattice = lattice
        self.solid = solid_mask  # (*grid), bool
        self.device = device
        self.q = lattice.q
        self.opp = lattice.opp.to(device)  # (Q,) opposite direction indices

        # Precompute: for each direction i, mask of fluid nodes where
        # x - e_i is solid (i.e., the population was streamed from a solid node)
        e = lattice.e.numpy().astype(int)
        ndim = solid_mask.dim()
        self.incoming_solid = []
        for i in range(self.q):
            shifted = solid_mask.clone()
            for d in range(ndim):
                s = int(e[i, d])
                if s != 0:
                    shifted = torch.roll(shifted, shifts=s, dims=d)
            # Fluid nodes with solid in the incoming direction
            self.incoming_solid.append(shifted & ~solid_mask)

    def apply(self, f_out: torch.Tensor, f_streamed: torch.Tensor) -> torch.Tensor:
        """Apply link bounce-back after streaming.

        For fluid nodes where streaming pulled from a solid neighbor,
        replace with the reflected outgoing population from the same node.

        Parameters
        ----------
        f_out : Tensor (Q, *grid), post-collision distributions
        f_streamed : Tensor (Q, *grid), post-streaming distributions

        Returns
        -------
        f_new : Tensor (Q, *grid)
        """
        f_new = f_streamed.clone()
        for i in range(self.q):
            mask = self.incoming_solid[i]
            if mask.any():
                # Replace population pulled from solid with reflected outgoing
                f_new[i][mask] = f_out[self.opp[i]][mask]
        return f_new


class WallBCs:
    """Combined wall boundary conditions."""

    def __init__(self, lattice, solid_mask, device="cuda"):
        self.bounce_back = BounceBack(lattice, solid_mask, device)

    def apply(self, f_pre, f_post):
        return self.bounce_back.apply(f_pre, f_post)
