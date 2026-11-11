//! Extended position-based cloth and tetrahedral volume constraints.
use crate::grid::SpatialHash;
use crate::particles::{boundary_targets, coupled_boundary, project_boundaries};
use crate::{PhysicsWorld, QueryFilter, Vec3};
use std::collections::BTreeSet;

mod surface;
use surface::SurfaceContacts;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SelfCollision {
    #[default]
    Nodes,
    Surface,
}

#[derive(Clone, Copy, Debug)]
struct Node {
    position: Vec3,
    previous: Vec3,
    velocity: Vec3,
    inverse_mass: f64,
}

#[derive(Clone, Debug)]
enum Constraint {
    Distance {
        indices: [usize; 2],
        rest: f64,
        compliance: f64,
        lambda: f64,
    },
    Volume {
        indices: [usize; 4],
        rest: f64,
        compliance: f64,
        lambda: f64,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct SoftSettings {
    pub gravity: Vec3,
    pub iterations: usize,
    pub damping: f64,
    pub radius: f64,
    /// Optional minimum separation for non-adjacent node pairs.
    pub self_collision_distance: Option<f64>,
    pub self_collision: SelfCollision,
}

impl Default for SoftSettings {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            iterations: 8,
            damping: 0.15,
            radius: 0.02,
            self_collision_distance: None,
            self_collision: SelfCollision::Nodes,
        }
    }
}

#[derive(Default)]
pub struct SoftBody {
    nodes: Vec<Node>,
    constraints: Vec<Constraint>,
    triangles: Vec<[usize; 3]>,
    adjacent: BTreeSet<(usize, usize)>,
    self_collision: Option<NodeBroadPhase>,
    surface_contacts: SurfaceContacts,
}

struct NodeBroadPhase {
    spacing: f64,
    grid: SpatialHash,
    positions: Vec<Vec3>,
}

impl NodeBroadPhase {
    fn solve(&mut self, nodes: &mut [Node], adjacent: &BTreeSet<(usize, usize)>) {
        self.positions.clear();
        self.positions.extend(nodes.iter().map(|n| n.position));
        self.grid.rebuild(&self.positions);
        let limit = self.spacing * self.spacing;
        for i in 0..nodes.len() {
            for j in self.grid.neighbors(self.positions[i]).filter(|&j| j > i) {
                let [a, b] = nodes.get_disjoint_mut([i, j]).expect("distinct nodes");
                let delta = b.position - a.position;
                let distance_squared = delta.length_squared();
                let total = a.inverse_mass + b.inverse_mass;
                if distance_squared >= limit || total == 0.0 || adjacent.contains(&(i, j)) {
                    continue;
                }
                let direction = delta.try_normalize().unwrap_or(Vec3::AXES[(i + j) % 3]);
                let change = direction * ((self.spacing - distance_squared.sqrt()) / total);
                a.position -= change * a.inverse_mass;
                b.position += change * b.inverse_mass;
            }
        }
    }
}

impl SoftBody {
    pub fn push_node(&mut self, position: Vec3, inverse_mass: f64) -> Option<usize> {
        if !position.is_finite() || !inverse_mass.is_finite() || inverse_mass < 0.0 {
            return None;
        }
        let index = self.nodes.len();
        self.nodes.push(Node {
            position,
            previous: position,
            velocity: Vec3::ZERO,
            inverse_mass,
        });
        Some(index)
    }

    pub fn position(&self, index: usize) -> Option<Vec3> {
        self.nodes.get(index).map(|node| node.position)
    }

    pub fn velocity(&self, index: usize) -> Option<Vec3> {
        self.nodes.get(index).map(|node| node.velocity)
    }

    pub fn positions(&self) -> impl ExactSizeIterator<Item = Vec3> + '_ {
        self.nodes.iter().map(|node| node.position)
    }

    pub fn triangles(&self) -> &[[usize; 3]] {
        &self.triangles
    }

    /// Adds a triangle to the deformable surface. Triangle edges are excluded
    /// from self-contact. Mechanical constraints can be added separately.
    pub fn add_triangle(&mut self, indices: [usize; 3]) -> bool {
        if !self.valid(indices, 0.0) {
            return false;
        }
        let [a, b, c] = indices.map(|i| self.nodes[i].position);
        if (b - a).cross(c - a).try_normalize().is_none() {
            return false;
        }
        for (a, b) in [
            (indices[0], indices[1]),
            (indices[1], indices[2]),
            (indices[2], indices[0]),
        ] {
            self.adjacent.insert(ordered(a, b));
        }
        self.triangles.push(indices);
        true
    }

    /// Sets a pinned point's position and zeroes its velocity.
    pub fn set_pin(&mut self, index: usize, position: Vec3) -> bool {
        if !position.is_finite() {
            return false;
        }
        let Some(node) = self.nodes.get_mut(index) else {
            return false;
        };
        *node = Node {
            position,
            previous: position,
            velocity: Vec3::ZERO,
            inverse_mass: 0.0,
        };
        true
    }

    pub fn add_distance(&mut self, indices: [usize; 2], compliance: f64) -> bool {
        if !self.valid(indices, compliance) {
            return false;
        }
        let [a, b] = indices.map(|i| self.nodes[i].position);
        let rest = a.distance(b);
        if !rest.is_finite() || rest <= 0.0 {
            return false;
        }
        self.adjacent.insert(ordered(indices[0], indices[1]));
        self.constraints.push(Constraint::Distance {
            indices,
            rest,
            compliance,
            lambda: 0.0,
        });
        true
    }

    pub fn add_volume(&mut self, indices: [usize; 4], compliance: f64) -> bool {
        if !self.valid(indices, compliance) {
            return false;
        }
        let rest = volume(indices.map(|i| self.nodes[i].position));
        if !rest.is_finite() || rest == 0.0 {
            return false;
        }
        for (i, &first) in indices.iter().enumerate() {
            for &second in &indices[i + 1..] {
                self.adjacent.insert(ordered(first, second));
            }
        }
        self.constraints.push(Constraint::Volume {
            indices,
            rest,
            compliance,
            lambda: 0.0,
        });
        true
    }

    fn valid<const N: usize>(&self, indices: [usize; N], compliance: f64) -> bool {
        compliance.is_finite()
            && compliance >= 0.0
            && indices
                .iter()
                .enumerate()
                .all(|(i, &index)| index < self.nodes.len() && !indices[..i].contains(&index))
    }

    /// Builds a grid in the XZ plane with edge, shear, and two-hop bend links.
    /// The first two corners of the grid are pinned.
    pub fn cloth(size: [usize; 2], spacing: f64, origin: Vec3, compliance: f64) -> Option<Self> {
        let [width, height] = size;
        if width < 2
            || height < 2
            || width.checked_mul(height).is_none()
            || !spacing.is_finite()
            || spacing <= 0.0
            || !compliance.is_finite()
            || compliance < 0.0
        {
            return None;
        }
        let mut cloth = Self::default();
        for y in 0..height {
            for x in 0..width {
                let mass = if y == 0 && (x == 0 || x == width - 1) {
                    0.0
                } else {
                    1.0
                };
                cloth.push_node(origin + spacing * Vec3::new(x as f64, 0.0, y as f64), mass)?;
            }
        }
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                for (dx, dy, factor) in [
                    (1, 0, 1.0),
                    (0, 1, 1.0),
                    (1, 1, 1.0),
                    (2, 0, 10.0),
                    (0, 2, 10.0),
                ] {
                    if x + dx < width
                        && y + dy < height
                        && !cloth.add_distance([i, (y + dy) * width + x + dx], compliance * factor)
                    {
                        return None;
                    }
                }
                if x + 1 < width && y + 1 < height {
                    if !cloth.add_distance([i + 1, i + width], compliance) {
                        return None;
                    }
                    if !cloth.add_triangle([i, i + width, i + 1])
                        || !cloth.add_triangle([i + 1, i + width, i + width + 1])
                    {
                        return None;
                    }
                }
            }
        }
        Some(cloth)
    }

    /// Advances XPBD, then exchanges normal impulses between movable nodes
    /// and rigid boundaries. Boundary projection occurs after internal XPBD
    /// constraints, and the caller advances the rigid world separately.
    pub fn step_coupled(
        &mut self,
        dt: f64,
        settings: SoftSettings,
        world: &mut PhysicsWorld,
        filter: QueryFilter,
    ) -> bool {
        if !self.step(dt, settings, None) {
            return false;
        }
        let targets = boundary_targets(world, filter);
        for node in self.nodes.iter_mut().filter(|n| n.inverse_mass > 0.0) {
            coupled_boundary(
                &mut node.position,
                &mut node.velocity,
                settings.radius,
                0.0,
                node.inverse_mass,
                world,
                &targets,
            );
        }
        true
    }

    /// Advances one XPBD step. Optional rigid colliders constrain node positions.
    pub fn step(
        &mut self,
        dt: f64,
        settings: SoftSettings,
        boundary: Option<(&PhysicsWorld, QueryFilter)>,
    ) -> bool {
        let dt_squared = dt * dt;
        if !dt.is_finite()
            || dt <= 0.0
            || !dt_squared.is_finite()
            || dt_squared == 0.0
            || !settings.gravity.is_finite()
            || settings.iterations == 0
            || !settings.damping.is_finite()
            || settings.damping < 0.0
            || !settings.radius.is_finite()
            || settings.radius < 0.0
            || settings
                .self_collision_distance
                .is_some_and(|d| !d.is_finite() || d <= 0.0)
        {
            return false;
        }
        if let Some(spacing) = settings.self_collision_distance
            && settings.self_collision == SelfCollision::Nodes
            && self
                .self_collision
                .as_ref()
                .is_none_or(|grid| grid.spacing != spacing)
        {
            self.self_collision = Some(NodeBroadPhase {
                spacing,
                grid: SpatialHash::new(spacing).expect("validated spacing"),
                positions: Vec::new(),
            });
        }
        let damping = (-settings.damping * dt).exp();
        for n in &mut self.nodes {
            n.previous = n.position;
            if n.inverse_mass > 0.0 {
                n.velocity = (n.velocity + settings.gravity * dt) * damping;
                n.position += n.velocity * dt;
            }
        }
        for constraint in &mut self.constraints {
            match constraint {
                Constraint::Distance { lambda, .. } | Constraint::Volume { lambda, .. } => {
                    *lambda = 0.0
                }
            }
        }
        for _ in 0..settings.iterations {
            for constraint in &mut self.constraints {
                constraint.solve(&mut self.nodes, dt_squared);
            }
            if let Some(thickness) = settings.self_collision_distance {
                match settings.self_collision {
                    SelfCollision::Nodes => self
                        .self_collision
                        .as_mut()
                        .expect("initialized node grid")
                        .solve(&mut self.nodes, &self.adjacent),
                    SelfCollision::Surface => self.surface_contacts.solve(
                        &mut self.nodes,
                        &self.triangles,
                        &self.adjacent,
                        thickness,
                    ),
                }
            }
            if let Some((world, filter)) = boundary {
                for node in self.nodes.iter_mut().filter(|n| n.inverse_mass > 0.0) {
                    project_boundaries(
                        &mut node.position,
                        settings.radius,
                        world,
                        filter,
                        |_, _, _| {},
                    );
                }
            }
        }
        for n in self.nodes.iter_mut().filter(|n| n.inverse_mass > 0.0) {
            n.velocity = (n.position - n.previous) / dt;
        }
        true
    }
}

fn ordered(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

fn volume([a, b, c, d]: [Vec3; 4]) -> f64 {
    (b - a).dot((c - a).cross(d - a)) / 6.0
}

impl Constraint {
    fn solve(&mut self, nodes: &mut [Node], dt_squared: f64) {
        match self {
            Self::Distance {
                indices,
                rest,
                compliance,
                lambda,
            } => {
                let [a, b] = indices.map(|i| nodes[i].position);
                let delta = b - a;
                let Some(direction) = delta.try_normalize() else {
                    return;
                };
                xpbd(
                    nodes,
                    *indices,
                    [-direction, direction],
                    delta.length() - *rest,
                    *compliance / dt_squared,
                    lambda,
                );
            }
            Self::Volume {
                indices,
                rest,
                compliance,
                lambda,
            } => {
                let points @ [a, b, c, d] = indices.map(|i| nodes[i].position);
                let gb = (c - a).cross(d - a) / 6.0;
                let gc = (d - a).cross(b - a) / 6.0;
                let gd = (b - a).cross(c - a) / 6.0;
                xpbd(
                    nodes,
                    *indices,
                    [-gb - gc - gd, gb, gc, gd],
                    volume(points) - *rest,
                    *compliance / dt_squared,
                    lambda,
                );
            }
        }
    }
}

fn xpbd<const N: usize>(
    nodes: &mut [Node],
    indices: [usize; N],
    gradients: [Vec3; N],
    error: f64,
    alpha: f64,
    lambda: &mut f64,
) {
    let k: f64 = indices
        .iter()
        .zip(gradients)
        .map(|(&i, gradient)| nodes[i].inverse_mass * gradient.length_squared())
        .sum();
    if k > 0.0 && alpha.is_finite() {
        let change = (-error - alpha * *lambda) / (k + alpha);
        *lambda += change;
        for (index, gradient) in indices.into_iter().zip(gradients) {
            let node = &mut nodes[index];
            node.position += gradient * (node.inverse_mass * change);
        }
    }
}
