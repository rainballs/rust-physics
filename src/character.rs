//! A kinematic sphere controller built on filtered swept queries.
use crate::spatial::{project, sweep_sphere};
use crate::{PhysicsWorld, QueryFilter, RayHit, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct SphereController {
    pub radius: f64,
    /// Separation left between the sphere and blocking surfaces.
    pub skin: f64,
    pub up: Vec3,
    /// Maximum ground angle from up, in radians.
    pub max_slope: f64,
    pub snap_distance: f64,
    pub max_slides: usize,
}

impl Default for SphereController {
    fn default() -> Self {
        Self {
            radius: 0.45,
            skin: 0.001,
            up: Vec3::Y,
            max_slope: 0.85,
            snap_distance: 0.1,
            max_slides: 5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CharacterMotion {
    pub position: Vec3,
    pub ground: Option<RayHit>,
    pub hits: Vec<RayHit>,
}

impl SphereController {
    /// Sweeps, slides, and optionally snaps to walkable ground. The controller
    /// returns a pose and leaves ownership of the character body to the caller.
    pub fn move_by(
        &self,
        world: &PhysicsWorld,
        mut position: Vec3,
        movement: Vec3,
        mut filter: QueryFilter,
    ) -> Option<CharacterMotion> {
        if !position.is_finite()
            || !movement.is_finite()
            || !self.up.is_normalized()
            || !self.radius.is_finite()
            || self.radius <= 0.0
            || !self.skin.is_finite()
            || self.skin < 0.0
            || self.skin >= self.radius
            || !self.snap_distance.is_finite()
            || self.snap_distance < 0.0
            || !(0.0..=std::f64::consts::FRAC_PI_2).contains(&self.max_slope)
            || self.max_slides == 0
        {
            return None;
        }
        filter.include_sensors = false;
        let slope = self.max_slope.cos();
        let mut hits = Vec::new();
        let mut ground = None;
        let mut planes: Vec<Vec3> = Vec::new();
        let mut remaining = movement;

        for _ in 0..self.max_slides {
            let overlap = world
                .bodies()
                .filter(|&(id, b)| filter.accepts(id, b))
                .filter_map(|(body, b)| {
                    let (point, normal, gap) = project(b, position);
                    let depth = self.radius - gap;
                    (depth > self.radius * 1e-9).then_some((
                        depth,
                        RayHit {
                            body,
                            position: point,
                            normal,
                            distance: 0.0,
                        },
                    ))
                })
                .max_by(|(a, ah), (b, bh)| a.total_cmp(b).then(bh.body.cmp(&ah.body)));
            let Some((depth, hit)) = overlap else {
                break;
            };
            position += hit.normal * (depth + self.skin);
            if hit.normal.dot(self.up) >= slope {
                ground = Some(hit);
            }
            hits.push(hit);
        }

        for _ in 0..self.max_slides {
            let length = remaining.length();
            if length <= self.radius * 1e-9 {
                break;
            }
            let direction = remaining / length;
            let Some(hit) = sweep(world, position, direction, self.radius, length, filter) else {
                position += remaining;
                break;
            };
            let travel = (hit.distance - self.skin).max(0.0);
            position += direction * travel;
            remaining -= direction * travel;
            if hit.normal.dot(self.up) >= slope {
                ground = Some(hit);
            }
            hits.push(hit);
            planes.push(hit.normal);

            // Project to a feasible direction for all recent contact planes.
            // At a two-plane corner, the intersection line is a candidate.
            remaining = clip_motion(remaining, &planes);
        }

        if movement.dot(self.up) <= 0.0
            && self.snap_distance > 0.0
            && let Some(hit) = sweep(
                world,
                position,
                -self.up,
                self.radius,
                self.snap_distance + self.skin,
                filter,
            )
            && hit.normal.dot(self.up) >= slope
        {
            position -= self.up * (hit.distance - self.skin).max(0.0);
            ground = Some(hit);
        }
        Some(CharacterMotion {
            position,
            ground,
            hits,
        })
    }
}

fn clip_motion(movement: Vec3, planes: &[Vec3]) -> Vec3 {
    let feasible = |candidate: Vec3| {
        planes
            .iter()
            .all(|n| candidate.dot(*n) >= -1e-12 * movement.length())
    };
    if feasible(movement) {
        return movement;
    }
    let mut best = Vec3::ZERO;
    let mut error = movement.length_squared();
    let mut consider = |candidate: Vec3| {
        let distance = candidate.distance_squared(movement);
        if distance < error && feasible(candidate) {
            best = candidate;
            error = distance;
        }
    };
    for (i, &first) in planes.iter().enumerate() {
        consider(movement - first * first.dot(movement));
        for &second in &planes[i + 1..] {
            if let Some(tangent) = first.cross(second).try_normalize() {
                consider(tangent * tangent.dot(movement));
            }
        }
    }
    best
}

fn sweep(
    world: &PhysicsWorld,
    origin: Vec3,
    direction: Vec3,
    radius: f64,
    max: f64,
    filter: QueryFilter,
) -> Option<RayHit> {
    world
        .bodies()
        .filter(|&(id, b)| filter.accepts(id, b))
        .filter_map(|(body, b)| {
            let (distance, normal) = sweep_sphere(b, origin, direction, radius, max)?;
            (direction.dot(normal) < -1e-10).then_some(RayHit {
                body,
                position: origin + direction * distance - normal * radius,
                normal,
                distance,
            })
        })
        .min_by(|a, b| a.distance.total_cmp(&b.distance).then(a.body.cmp(&b.body)))
}

/// Swept capsule movement with sliding, ground snapping, and step climbing.
#[derive(Clone, Copy, Debug)]
pub struct CapsuleController {
    pub radius: f64,
    pub half_segment: f64,
    pub skin: f64,
    pub up: Vec3,
    pub max_slope: f64,
    pub step_height: f64,
    pub snap_distance: f64,
    pub max_slides: usize,
}

impl Default for CapsuleController {
    fn default() -> Self {
        Self {
            radius: 0.3,
            half_segment: 0.4,
            skin: 0.001,
            up: Vec3::Y,
            max_slope: 0.85,
            step_height: 0.35,
            snap_distance: 0.1,
            max_slides: 5,
        }
    }
}

impl CapsuleController {
    pub fn move_by(
        &self,
        world: &PhysicsWorld,
        position: Vec3,
        movement: Vec3,
        mut filter: QueryFilter,
    ) -> Option<CharacterMotion> {
        use crate::{Quat, RigidBodyBuilder, Shape};
        if !position.is_finite()
            || !movement.is_finite()
            || !self.up.is_normalized()
            || !self.radius.is_finite()
            || self.radius <= 0.0
            || !self.half_segment.is_finite()
            || self.half_segment < 0.0
            || !self.skin.is_finite()
            || self.skin <= 0.0
            || self.skin >= self.radius
            || !self.step_height.is_finite()
            || self.step_height < 0.0
            || !self.snap_distance.is_finite()
            || self.snap_distance < 0.0
            || !(0.0..=std::f64::consts::FRAC_PI_2).contains(&self.max_slope)
            || self.max_slides == 0
        {
            return None;
        }
        filter.include_sensors = false;
        let rotation = Quat::from_rotation_arc(Vec3::Y, self.up);
        let mut capsule = RigidBodyBuilder::kinematic()
            .shape(Shape::capsule(self.radius, self.half_segment))
            .position(position)
            .rotation(rotation)
            .try_build()
            .ok()?;
        let slope = self.max_slope.cos();
        let mut hits = Vec::new();
        let mut ground = None;
        let mut geometry = Vec::with_capacity(8);
        for _ in 0..self.max_slides {
            let mut best = None;
            for (id, collider) in world.bodies().filter(|&(id, b)| filter.accepts(id, b)) {
                geometry.clear();
                crate::collision::contacts(collider, &capsule, &mut geometry);
                for contact in &geometry {
                    if contact.depth > self.skin * 1e-3
                        && best.is_none_or(|(depth, previous, _)| {
                            contact.depth > depth || (contact.depth == depth && id < previous)
                        })
                    {
                        best = Some((contact.depth, id, *contact));
                    }
                }
            }
            let Some((depth, body, contact)) = best else {
                break;
            };
            capsule.position += contact.normal * (depth + self.skin);
            let hit = RayHit {
                body,
                position: contact.point,
                normal: contact.normal,
                distance: 0.0,
            };
            if hit.normal.dot(self.up) >= slope {
                ground = Some(hit);
            }
            hits.push(hit);
        }

        let mut grounded = ground.is_some()
            || self
                .floor(world, &capsule, self.snap_distance + self.skin, filter)
                .is_some();
        let mut remaining = movement;
        let mut normals = Vec::new();
        for _ in 0..self.max_slides {
            let distance = remaining.length();
            if distance <= self.radius * 1e-9 {
                break;
            }
            let direction = remaining / distance;
            let Some(hit) = sweep_capsule(world, &capsule, direction, distance, filter) else {
                capsule.position += remaining;
                break;
            };
            let travel = (hit.distance - self.skin).max(0.0);
            capsule.position += direction * travel;
            remaining -= direction * travel;
            let walkable = hit.normal.dot(self.up) >= slope;
            if walkable {
                ground = Some(hit);
                grounded = true;
            }
            hits.push(hit);
            if !walkable
                && grounded
                && movement.dot(self.up) <= 0.0
                && let Some((position, landing)) = self.try_step(world, &capsule, remaining, filter)
            {
                capsule.position = position;
                ground = Some(landing);
                break;
            }
            normals.push(hit.normal);
            remaining = clip_motion(remaining, &normals);
        }

        if movement.dot(self.up) <= 0.0
            && let Some(hit) = self.floor(world, &capsule, self.snap_distance + self.skin, filter)
        {
            capsule.position -= self.up * (hit.distance - self.skin).max(0.0);
            ground = Some(hit);
        }
        capsule.position.is_finite().then_some(CharacterMotion {
            position: capsule.position,
            ground,
            hits,
        })
    }

    fn floor(
        &self,
        world: &PhysicsWorld,
        capsule: &crate::RigidBody,
        range: f64,
        filter: QueryFilter,
    ) -> Option<RayHit> {
        sweep_capsule(world, capsule, -self.up, range, filter)
            .filter(|hit| hit.normal.dot(self.up) >= self.max_slope.cos())
    }

    fn try_step(
        &self,
        world: &PhysicsWorld,
        capsule: &crate::RigidBody,
        movement: Vec3,
        filter: QueryFilter,
    ) -> Option<(Vec3, RayHit)> {
        if self.step_height <= 0.0
            || sweep_capsule(
                world,
                capsule,
                self.up,
                self.step_height + self.skin,
                filter,
            )
            .is_some()
        {
            return None;
        }
        let horizontal = movement - self.up * movement.dot(self.up);
        let direction = horizontal.try_normalize()?;
        let distance = horizontal.length();
        let mut raised = capsule.clone();
        raised.position += self.up * self.step_height;
        let travel = sweep_capsule(world, &raised, direction, distance, filter)
            .map_or(distance, |hit| (hit.distance - self.skin).max(0.0));
        if travel <= self.skin {
            return None;
        }
        raised.position += direction * travel;
        let landing = self.floor(
            world,
            &raised,
            self.step_height + self.snap_distance + self.skin,
            filter,
        )?;
        let position = raised.position - self.up * (landing.distance - self.skin).max(0.0);
        ((position - capsule.position).dot(self.up) <= self.step_height + self.skin)
            .then_some((position, landing))
    }
}

fn sweep_capsule(
    world: &PhysicsWorld,
    capsule: &crate::RigidBody,
    direction: Vec3,
    max: f64,
    filter: QueryFilter,
) -> Option<RayHit> {
    world
        .bodies()
        .filter(|&(id, b)| filter.accepts(id, b))
        .filter_map(|(body, target)| {
            let (distance, normal, position) =
                crate::spatial::cast_convex_body(capsule, target, direction, max)?;
            (direction.dot(normal) < -1e-10).then_some(RayHit {
                body,
                position,
                normal,
                distance,
            })
        })
        .min_by(|a, b| a.distance.total_cmp(&b.distance).then(a.body.cmp(&b.body)))
}
