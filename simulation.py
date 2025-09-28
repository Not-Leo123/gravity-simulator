"""
simulation.py

Physics core for the gravity simulator.

Provides:
- Body class (pos/vel/mass/radius/color/type)
- accel: compute accelerations (softened Newtonian gravity)
- velocity_verlet_step: integrate bodies in-place
- merge_bodies: merge two bodies conserving momentum
- handle_collisions: merge overlapping bodies (including black-hole absorption)
- trigger_supernova: remove a massive star and return fragments + remnant
"""

from typing import List, Tuple
import numpy as np

G = 9.8
SOFTENING = 1e-3


class Body:
    def __init__(self,
                 position,
                 velocity,
                 mass,
                 radius=0.05,
                 color="cyan",
                 body_type="planet"):
        """
        position, velocity : 2-element iterables (world units)
        mass : scalar
        radius : world units (not pixels)
        color : string for GUI
        body_type : "planet", "star", "blackhole"
        """
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = float(mass)
        self.radius = float(radius)
        self.color = color
        self.body_type = body_type
        # GUI may attach .id and .trail attributes; physics doesn't touch them

    def __repr__(self):
        return f"<Body type={self.body_type} mass={self.mass:.3g} pos={self.position}>"

# --- Physics utilities ---


def accel(positions: np.ndarray, masses: np.ndarray, G_const: float = G, eps: float = SOFTENING) -> np.ndarray:
    """
    positions: (N,2) array
    masses: (N,) array
    returns: (N,2) accelerations
    """
    positions = np.asarray(positions, dtype=float)
    masses = np.asarray(masses, dtype=float)
    N = positions.shape[0]
    if N == 0:
        return np.zeros((0, 2), dtype=float)

    # r_ij = r_j - r_i
    r = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]  # (N,N,2)
    dist2 = np.sum(r * r, axis=2) + eps**2                         # (N,N)
    inv_dist3 = dist2 ** (-1.5)
    np.fill_diagonal(inv_dist3, 0.0)
    a = G_const * np.sum(r * inv_dist3[:, :, np.newaxis] * masses[np.newaxis, :, np.newaxis], axis=1)
    return a


def velocity_verlet_step(bodies: List[Body], dt: float):
    """
    Advance `bodies` in place by one velocity-Verlet step.
    """
    if not bodies:
        return

    positions = np.array([b.position for b in bodies], dtype=float)
    velocities = np.array([b.velocity for b in bodies], dtype=float)
    masses = np.array([b.mass for b in bodies], dtype=float)

    a = accel(positions, masses)
    velocities_half = velocities + 0.5 * a * dt
    positions_new = positions + velocities_half * dt
    a_new = accel(positions_new, masses)
    velocities_new = velocities_half + 0.5 * a_new * dt

    for b, pos, vel in zip(bodies, positions_new, velocities_new):
        b.position = pos
        b.velocity = vel


def merge_bodies(b1: Body, b2: Body) -> Body:
    """
    Merge two bodies conserving momentum. Return new Body.
    Body type/color chosen from more massive one (except special rules could be added).
    """
    total_mass = b1.mass + b2.mass
    pos = (b1.position * b1.mass + b2.position * b2.mass) / total_mass
    vel = (b1.velocity * b1.mass + b2.velocity * b2.mass) / total_mass
    # combine radii area-like: sqrt(r1^2 + r2^2)
    radius = float((b1.radius ** 2 + b2.radius ** 2) ** 0.5)
    # prefer heavier body's type/color (blackhole dominates)
    if b1.body_type == "blackhole" or b2.body_type == "blackhole":
        body_type = "blackhole"
    else:
        body_type = b1.body_type if b1.mass >= b2.mass else b2.body_type
    color = b1.color if b1.mass >= b2.mass else b2.color
    return Body(pos, vel, total_mass, radius=radius, color=color, body_type=body_type)


def handle_collisions(bodies: List[Body]) -> Tuple[List[Body], List[Tuple[Body, str]]]:
    """
    Detect overlapping bodies and merge them.
    Returns (new_bodies, flashes) where flashes is list of (merged_body, final_color)
    for GUI to flash the merged body then set to final_color.
    Black holes instantly absorb colliding bodies and remain blackhole type.
    Radii compared in world units (b.radius + b2.radius).
    """
    N = len(bodies)
    if N <= 1:
        return bodies[:], []

    merged_flags = set()
    new_bodies: List[Body] = []
    flashes: List[Tuple[Body, str]] = []

    for i in range(N):
        if i in merged_flags:
            continue
        bi = bodies[i]
        did_merge = False
        for j in range(i + 1, N):
            if j in merged_flags:
                continue
            bj = bodies[j]
            dist = np.linalg.norm(bi.position - bj.position)
            if dist < (bi.radius + bj.radius):
                # If either is blackhole -> absorb (result is blackhole)
                if bi.body_type == "blackhole" or bj.body_type == "blackhole":
                    # blackhole absorbs other -> merge and result is blackhole
                    merged = merge_bodies(bi, bj)
                    merged.body_type = "blackhole"
                    merged.color = "black"
                else:
                    merged = merge_bodies(bi, bj)
                new_bodies.append(merged)
                # use heavier body's color as final color for flash fade
                final_color = bi.color if bi.mass >= bj.mass else bj.color
                flashes.append((merged, final_color))
                merged_flags.update({i, j})
                did_merge = True
                break
        if not did_merge:
            new_bodies.append(bi)

    return new_bodies, flashes


def trigger_supernova(bodies: List[Body], star: Body) -> List[Body]:
    """
    Explode `star` if it is massive enough; remove it from `bodies` and
    return a list of new Body objects: fragments + remnant (blackhole/neutron).
    If star.mass below threshold, returns [] and does nothing.
    """
    MASS_THRESHOLD = 20.0
    if star.mass < MASS_THRESHOLD:
        return []

    # remove star from list (caller should handle GUI cleanup)
    try:
        bodies.remove(star)
    except ValueError:
        # already removed
        pass

    pos = star.position.copy()
    vel = star.velocity.copy()

    fragments: List[Body] = []
    n_frags = np.random.randint(6, 12)
    for _ in range(n_frags):
        m = float(np.random.uniform(0.05, 1.5))
        v = vel + np.random.uniform(-2.0, 2.0, size=2)
        offset = np.random.uniform(-0.5, 0.5, size=2)
        color = str(np.random.choice(["orange", "red", "yellow", "white"]))
        frag = Body(pos + offset, v, m, radius=max(0.02, (m ** 0.5) * 0.03), color=color, body_type="planet")
        fragments.append(frag)

    # remnant (most likely black hole)
    if np.random.rand() < 0.75:
        remnant = Body(pos, vel, star.mass * 0.4, radius=max(0.05, star.radius * 0.6), color="darkgray", body_type="blackhole")
    else:
        remnant = Body(pos, vel, star.mass * 0.2, radius=max(0.04, star.radius * 0.5), color="blue", body_type="star")

    return fragments + [remnant]
