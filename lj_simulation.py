"""Small, student-friendly 2-D Lennard-Jones NVE simulator.

The notebook intentionally exposes only a few high-level operations. The
force calculation, periodic boundaries, initialization, and velocity-Verlet
integrator live here so that the computational experiment remains the focus.

All quantities use reduced Lennard-Jones units: epsilon = sigma = m = k_B = 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def minimum_image(displacements: np.ndarray, box_length: float) -> np.ndarray:
    """Apply the minimum-image convention to one or more displacement vectors."""
    return displacements - box_length * np.rint(displacements / box_length)


def observation_count(positions: np.ndarray, box_length: float, fraction: float = 0.25) -> int:
    """Return the number of particles in a centered square observation window.

    ``fraction`` is the fraction of the box side occupied by the window, so the
    window's area fraction is approximately ``fraction**2``.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    centered = (positions - box_length / 2.0) % box_length - box_length / 2.0
    half_width = 0.5 * fraction * box_length
    inside = np.all(np.abs(centered) <= half_width, axis=1)
    return int(np.count_nonzero(inside))


@dataclass
class LJSystem:
    """A two-dimensional Lennard-Jones system evolved in the NVE ensemble."""

    N: int = 36
    rho: float = 0.6
    temperature: float = 1.0
    dt: float = 0.003
    rcut: float = 2.5
    seed: Optional[int] = 7
    positions: Optional[np.ndarray] = None
    velocities: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.N < 2:
            raise ValueError("N must be at least 2")
        if self.rho <= 0 or self.dt <= 0:
            raise ValueError("rho and dt must be positive")
        if self.rcut <= 0:
            raise ValueError("rcut must be positive")

        self.rng = np.random.default_rng(self.seed)
        self.box_length = float(np.sqrt(self.N / self.rho))

        if self.positions is None:
            self.positions = self._initial_positions()
        else:
            self.positions = np.asarray(self.positions, dtype=float).copy() % self.box_length
        if self.positions.shape != (self.N, 2):
            raise ValueError("positions must have shape (N, 2)")

        if self.velocities is None:
            self.velocities = self._initial_velocities(self.temperature)
        else:
            self.velocities = np.asarray(self.velocities, dtype=float).copy()
        if self.velocities.shape != (self.N, 2):
            raise ValueError("velocities must have shape (N, 2)")

        self.forces, self.potential_energy = self._forces_and_potential(self.positions)

    def _initial_positions(self) -> np.ndarray:
        """Place particles on a gently jittered square lattice."""
        nside = int(np.ceil(np.sqrt(self.N)))
        spacing = self.box_length / nside
        grid = np.array(
            [(i * spacing + 0.5 * spacing, j * spacing + 0.5 * spacing)
             for i in range(nside) for j in range(nside)],
            dtype=float,
        )[: self.N]
        jitter = self.rng.normal(0.0, 0.055 * spacing, size=(self.N, 2))
        return (grid + jitter) % self.box_length

    def _initial_velocities(self, temperature: float) -> np.ndarray:
        velocities = self.rng.normal(0.0, np.sqrt(max(temperature, 1e-12)), size=(self.N, 2))
        velocities -= velocities.mean(axis=0)
        return self._rescale_to_kinetic_energy(velocities, self.target_kinetic_energy(temperature))

    def target_kinetic_energy(self, temperature: float) -> float:
        """Equipartition target after removing the center-of-mass motion."""
        degrees_of_freedom = 2 * self.N - 2
        return 0.5 * degrees_of_freedom * temperature

    @staticmethod
    def _rescale_to_kinetic_energy(velocities: np.ndarray, kinetic_energy: float) -> np.ndarray:
        if kinetic_energy < 0:
            raise ValueError("requested kinetic energy must be nonnegative")
        velocities = np.asarray(velocities, dtype=float).copy()
        velocities -= velocities.mean(axis=0)
        current = 0.5 * np.sum(velocities * velocities)
        if kinetic_energy == 0:
            return np.zeros_like(velocities)
        if current <= 0:
            raise ValueError("cannot rescale zero velocities to positive kinetic energy")
        return velocities * np.sqrt(kinetic_energy / current)

    def set_kinetic_energy(self, kinetic_energy: float) -> None:
        """Rescale velocities once, e.g. while preparing an equal-energy replica."""
        self.velocities = self._rescale_to_kinetic_energy(self.velocities, kinetic_energy)

    def kinetic_energy(self) -> float:
        return float(0.5 * np.sum(self.velocities * self.velocities))

    def total_energy(self) -> float:
        return self.kinetic_energy() + self.potential_energy

    def _forces_and_potential(self, positions: np.ndarray) -> tuple[np.ndarray, float]:
        forces = np.zeros_like(positions)
        cutoff_shift = 4.0 * (self.rcut ** -12 - self.rcut ** -6)
        delta = minimum_image(positions[:, None, :] - positions[None, :, :], self.box_length)
        distances_squared = np.sum(delta * delta, axis=2)
        pair_mask = np.triu(np.ones((self.N, self.N), dtype=bool), k=1)
        pair_mask &= distances_squared < self.rcut ** 2
        pair_mask &= distances_squared > 1e-12
        i_indices, j_indices = np.where(pair_mask)
        if len(i_indices) == 0:
            return forces, 0.0
        r2 = distances_squared[i_indices, j_indices]
        vectors = delta[i_indices, j_indices]
        inv_r2 = 1.0 / r2
        inv_r6 = inv_r2 ** 3
        inv_r12 = inv_r6 ** 2
        potential = float(np.sum(4.0 * (inv_r12 - inv_r6) - cutoff_shift))
        force_factor = 24.0 * inv_r2 * (2.0 * inv_r12 - inv_r6)
        pair_forces = force_factor[:, None] * vectors
        np.add.at(forces, i_indices, pair_forces)
        np.add.at(forces, j_indices, -pair_forces)
        return forces, potential

    def step(self) -> None:
        """Advance one velocity-Verlet step with no thermostat."""
        self.velocities += 0.5 * self.dt * self.forces
        self.positions = (self.positions + self.dt * self.velocities) % self.box_length
        self.forces, self.potential_energy = self._forces_and_potential(self.positions)
        self.velocities += 0.5 * self.dt * self.forces

    def run(self, steps: int, sample_interval: int = 1, store_positions: bool = True) -> dict[str, np.ndarray]:
        """Run NVE dynamics and return sampled time series.

        The returned dictionary contains ``time``, ``kinetic``, ``potential``,
        ``total``, and, when requested, ``positions``.
        """
        if steps < 0 or sample_interval < 1:
            raise ValueError("steps must be nonnegative and sample_interval must be positive")
        times, kinetic, potential, total, frames = [], [], [], [], []
        for step_number in range(steps + 1):
            if step_number % sample_interval == 0:
                k = self.kinetic_energy()
                u = self.potential_energy
                times.append(step_number * self.dt)
                kinetic.append(k)
                potential.append(u)
                total.append(k + u)
                if store_positions:
                    frames.append(self.positions.copy())
            if step_number < steps:
                self.step()
        result = {
            "time": np.asarray(times),
            "kinetic": np.asarray(kinetic),
            "potential": np.asarray(potential),
            "total": np.asarray(total),
        }
        if store_positions:
            result["positions"] = np.asarray(frames)
        return result


def make_equal_energy_replicas(
    reference: LJSystem,
    count: int = 48,
    seed: int = 100,
    target_energy: Optional[float] = None,
) -> list[LJSystem]:
    """Create independent systems with the same initial total energy.

    Each replica receives an independently initialized configuration. Its
    velocities are rescaled so that ``K + U`` equals the reference energy to
    floating-point precision before NVE evolution begins.
    """
    if count < 1:
        raise ValueError("count must be positive")
    target = reference.total_energy() if target_energy is None else float(target_energy)
    replicas: list[LJSystem] = []
    replica_seed = int(seed)
    attempts = 0
    while len(replicas) < count and attempts < 20 * count:
        attempts += 1
        candidate = LJSystem(
            N=reference.N,
            rho=reference.rho,
            temperature=reference.temperature,
            dt=reference.dt,
            rcut=reference.rcut,
            seed=replica_seed,
        )
        replica_seed += 1
        needed_kinetic = target - candidate.potential_energy
        if needed_kinetic <= 1e-8:
            continue
        candidate.set_kinetic_energy(needed_kinetic)
        replicas.append(candidate)
    if len(replicas) != count:
        raise RuntimeError("could not construct the requested equal-energy replicas")
    return replicas
