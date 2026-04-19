"""Lattice definitions for LBM: D2Q9 and D3Q19.

Stores velocity vectors, weights, opposite direction indices, and
speed of sound as PyTorch tensors (float32, GPU-ready).
"""
import torch
import numpy as np


class Lattice:
    """Base lattice class."""

    def __init__(self, e: np.ndarray, w: np.ndarray, opp: np.ndarray, cs2: float = 1 / 3):
        self.dim = e.shape[1]
        self.q = e.shape[0]
        self.cs2 = cs2  # speed of sound squared

        # Velocity vectors: (Q, dim)
        self.e = torch.tensor(e, dtype=torch.float32)
        # Weights: (Q,)
        self.w = torch.tensor(w, dtype=torch.float32)
        # Opposite direction indices: (Q,)
        self.opp = torch.tensor(opp, dtype=torch.long)

        # Precompute e_i * w_i for equilibrium: (Q, dim)
        self.ew = self.e * self.w.unsqueeze(1)

        # Precompute e_i^2 for equilibrium: (Q,)
        self.e_sq = (self.e ** 2).sum(dim=1)

        # Streaming shifts: e[i] gives the shift for population i
        # Negative because we pull from upstream
        self.stream_shifts = -self.e.numpy().astype(int)


class D2Q9(Lattice):
    """D2Q9 lattice for 2D simulations.

    Velocities:
        0: rest
        1-4: cardinal (E, W, N, S)
        5-8: diagonal (NE, NW, SW, SE)
    """

    def __init__(self):
        e = np.array([
            [0, 0],
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [1, 1], [-1, 1], [-1, -1], [1, -1],
        ], dtype=np.int32)
        w = np.array([
            4 / 9,
            1 / 9, 1 / 9, 1 / 9, 1 / 9,
            1 / 36, 1 / 36, 1 / 36, 1 / 36,
        ], dtype=np.float32)
        opp = np.array([0, 2, 1, 4, 3, 7, 8, 5, 6], dtype=np.int64)
        super().__init__(e, w, opp)


class D3Q19(Lattice):
    """D3Q19 lattice for 3D simulations.

    Velocities:
        0: rest
        1-6:  face (±x, ±y, ±z)
        7-18: edge (plane diagonals)
    """

    def __init__(self):
        e = np.array([
            [0, 0, 0],
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
            [1, 1, 0], [-1, 1, 0],
            [-1, -1, 0], [1, -1, 0],
            [1, 0, 1], [-1, 0, 1],
            [-1, 0, -1], [1, 0, -1],
            [0, 1, 1], [0, -1, 1],
            [0, -1, -1], [0, 1, -1],
        ], dtype=np.int32)
        w = np.array([
            1 / 3,
            1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18,
            1 / 36, 1 / 36, 1 / 36, 1 / 36,
            1 / 36, 1 / 36, 1 / 36, 1 / 36,
            1 / 36, 1 / 36, 1 / 36, 1 / 36,
        ], dtype=np.float32)
        opp = np.array([
            0,
            2, 1, 4, 3, 6, 5,
            9, 10, 7, 8,
            13, 14, 11, 12,
            17, 18, 15, 16,
        ], dtype=np.int64)
        super().__init__(e, w, opp)


class D3Q27(Lattice):
    """D3Q27 lattice for 3D simulations.

    Includes 8 corner velocities in addition to D3Q19, providing
    full 3rd-order isotropy. Required for the Lee & Liu (2010)
    free-energy LBM where the 26-point gradient stencils depend
    on all 27 velocity directions.

    Velocities:
        0:    rest
        1-6:   face (±x, ±y, ±z)
        7-18:  edge (plane diagonals)
        19-26: corner (space diagonals)

    Weights:
        rest:   8/27
        face:   2/27
        edge:   1/54
        corner: 1/216
    """

    def __init__(self):
        e = np.array([
            # 0: rest
            [0, 0, 0],
            # 1-6: face
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
            # 7-18: edge (plane diagonals)
            [1, 1, 0], [-1, 1, 0],
            [-1, -1, 0], [1, -1, 0],
            [1, 0, 1], [-1, 0, 1],
            [-1, 0, -1], [1, 0, -1],
            [0, 1, 1], [0, -1, 1],
            [0, -1, -1], [0, 1, -1],
            # 19-26: corner (space diagonals)
            [1, 1, 1], [1, 1, -1],
            [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1],
            [-1, -1, 1], [-1, -1, -1],
        ], dtype=np.int32)
        w = np.array([
            # 0: rest
            8.0 / 27.0,
            # 1-6: face
            2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0,
            # 7-18: edge
            1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0,
            1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0,
            1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0,
            # 19-26: corner
            1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0,
            1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0,
        ], dtype=np.float32)
        opp = np.array([
            0,
            2, 1, 4, 3, 6, 5,   # face
            9, 10, 7, 8,        # edge
            11, 12, 13, 14,      # edge
            15, 16, 17, 18,      # edge
            26, 25, 24, 23,      # corner (opposite of 19-22)
            22, 21, 20, 19,      # corner (opposite of 23-26)
        ], dtype=np.int64)
        super().__init__(e, w, opp)
