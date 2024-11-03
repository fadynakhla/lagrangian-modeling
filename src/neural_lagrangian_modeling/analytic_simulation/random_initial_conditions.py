import numpy as np
from typing import Optional, Tuple
from neural_lagrangian_modeling import datamodels

def random_initial_conditions(
    dims: int = 2,
    mass_range: tuple[float, float] = (0.1, 10.0),
    radius_range: tuple[float, float] = (0.5, 2.0),
    velocity_scale: float = 1.0,
    seed: Optional[int] = None
) -> tuple[datamodels.MassiveBody, datamodels.MassiveBody, datamodels.MassiveBody]:
    """Generate physically-motivated random initial conditions.

    Args:
        dims: Number of dimensions (2 or 3)
        mass_range: (min, max) masses
        radius_range: (min, max) initial separation from center of mass
        velocity_scale: Scale factor for velocities (1.0 = roughly circular orbits)
        seed: Random seed for reproducibility

    The bodies are initialized with:
    - Random masses within mass_range
    - Positions roughly evenly distributed in space (not too close)
    - Velocities that give approximately circular/elliptical orbits
    - Center of mass at origin
    - Total linear momentum zero
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random masses
    masses = np.random.uniform(*mass_range, 3)
    total_mass = np.sum(masses)

    # Generate positions that aren't too close together
    positions = []
    while len(positions) < 3:
        # Generate random angles
        if dims == 2:
            theta = np.random.uniform(0, 2*np.pi)
            pos = np.array([
                np.cos(theta),
                np.sin(theta)
            ], dtype=np.float128)
        else:
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            pos = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ], dtype=np.float128)

        # Random radius within range
        r = np.random.uniform(*radius_range)
        pos *= r

        # Check if not too close to other positions
        min_separation = (radius_range[1] - radius_range[0]) / 4
        if all(np.linalg.norm(pos - p) > min_separation for p in positions):
            positions.append(pos)

    positions = np.array(positions, dtype=np.float128)

    # Center the positions at origin
    com = np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass
    positions -= com

    # Generate velocities for approximately circular orbits
    velocities = []
    for i, (pos, mass) in enumerate(zip(positions, masses)):
        # Calculate approximate orbital velocity considering other masses
        other_masses = np.delete(masses, i)
        other_positions = np.delete(positions, i, axis=0)

        # Get approximate central force
        r = np.linalg.norm(pos)
        if r > 0:
            # Calculate velocity perpendicular to position vector
            if dims == 2:
                # Rotate position vector 90 degrees
                vel_direction = np.array([-pos[1], pos[0]], dtype=np.float128)
            else:
                # Cross product with arbitrary vector (avoid zero velocity)
                ref = np.array([0, 0, 1], dtype=np.float128)
                if np.abs(np.dot(pos, ref)) > 0.9:
                    ref = np.array([0, 1, 0], dtype=np.float128)
                vel_direction = np.cross(pos, ref)

            vel_direction /= np.linalg.norm(vel_direction)

            # Velocity magnitude for circular-ish orbit
            v_mag = np.sqrt(total_mass / r) * velocity_scale
            velocities.append(vel_direction * v_mag)
        else:
            velocities.append(np.zeros(3, dtype=np.float128))

    velocities = np.array(velocities, dtype=np.float128)

    # Ensure center of mass velocity is zero
    com_vel = np.sum(velocities * masses[:, np.newaxis], axis=0) / total_mass
    velocities -= com_vel

    # Create MassiveBody objects
    bodies = tuple(
        datamodels.MassiveBody(
            mass=float(mass),
            position=pos,
            velocity=vel
        )
        for mass, pos, vel in zip(masses, positions, velocities)
    )

    return bodies

def get_random_simulation_params(dims: int = 2) -> dict:
    """Get recommended simulation parameters for random initial conditions."""
    return {
        'dt': 0.001,
        'steps': 10000,
        'dims': dims,
        'trail_length': 500,
        'interval': 20
    }
