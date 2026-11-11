use rust_physics::terrain::TriangleMesh;
use rust_physics::{PhysicsWorld, Quat, QueryFilter, RigidBodyBuilder, Shape, Vec3};
use std::sync::Arc;

#[test]
fn heightfield_queries_use_triangle_surface_and_body_transform() {
    let mesh =
        Arc::new(TriangleMesh::heightfield([2, 2], [1.0, 1.0], &[0.0, 0.5, 0.0, 0.5]).unwrap());
    let q = Quat::from_rotation_y(0.5);
    let origin = Vec3::new(2.0, 3.0, 4.0);
    let mut world = PhysicsWorld::new();
    let terrain = world.add_body(
        RigidBodyBuilder::static_body()
            .position(origin)
            .rotation(q)
            .shape(Shape::Mesh(mesh.clone()))
            .build(),
    );
    assert!(
        RigidBodyBuilder::dynamic()
            .shape(Shape::Mesh(mesh))
            .try_build()
            .is_err()
    );
    let p = origin + q * Vec3::new(0.5, 3.0, 0.5);
    let filter = QueryFilter::default();
    let hit = world.raycast(p, -Vec3::Y, 5.0, filter).unwrap();
    let normal = q * Vec3::new(-0.5, 1.0, 0.0).normalize();
    assert_eq!(hit.body, terrain);
    assert!((hit.distance - 2.75).abs() < 1e-12);
    assert!(hit.normal.distance(normal) < 1e-12);
    let near = world
        .nearest_surface(hit.position + normal * 0.1, filter)
        .unwrap();
    assert!((near.distance - 0.1).abs() < 1e-12);
    assert!(near.position.distance(hit.position) < 1e-12);
    assert_eq!(
        world
            .overlap_sphere(hit.position, 0.05, filter)
            .collect::<Vec<_>>(),
        [terrain]
    );
}

#[test]
fn sphere_settles_on_a_triangulation_seam() {
    let mesh = Arc::new(TriangleMesh::heightfield([4, 4], [1.0, 1.0], &[0.0; 16]).unwrap());
    let mut world = PhysicsWorld::new();
    world.add_body(
        RigidBodyBuilder::static_body()
            .shape(Shape::Mesh(mesh))
            .build(),
    );
    let sphere = world.add_body(
        RigidBodyBuilder::dynamic()
            .position(Vec3::new(1.5, 2.0, 1.5))
            .build(),
    );
    for _ in 0..180 {
        world.step_fixed().unwrap();
    }
    let b = world.body(sphere).unwrap();
    assert!((b.position().y - 0.5).abs() < 0.003);
    assert!((b.position().x - 1.5).abs() < 1e-12);
    assert!((b.position().z - 1.5).abs() < 1e-12);
    assert!(b.is_sleeping());
}

#[test]
fn mesh_ccd_detects_crossing_and_cast_reaches_an_exposed_edge() {
    let mesh = Arc::new(
        TriangleMesh::new(
            vec![Vec3::ZERO, Vec3::Z * 2.0, Vec3::X * 2.0],
            vec![[0, 1, 2]],
        )
        .unwrap(),
    );
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO).unwrap();
    let terrain = world.add_body(
        RigidBodyBuilder::static_body()
            .shape(Shape::Mesh(mesh))
            .build(),
    );
    let filter = QueryFilter::default();
    let hit = world
        .cast_sphere(Vec3::new(-2.0, 0.4, 1.0), 0.5, Vec3::X, 5.0, filter)
        .unwrap();
    assert_eq!(hit.body, terrain);
    assert!((hit.distance - 1.7).abs() < 1e-12);
    assert!(hit.normal.distance(Vec3::new(-0.6, 0.8, 0.0)) < 1e-12);
    let id = world.add_body(
        RigidBodyBuilder::dynamic()
            .position(Vec3::new(0.5, 3.0, 0.5))
            .velocity(-Vec3::Y * 1_000.0)
            .ccd(true)
            .build(),
    );
    assert_eq!(world.step_fixed().unwrap().ccd_impacts, 1);
    assert!(world.body(id).unwrap().position().y >= 0.5);
}

#[test]
fn malformed_mesh_inputs_are_rejected() {
    assert!(TriangleMesh::heightfield([2, 2], [1.0, 1.0], &[0.0; 3]).is_none());
    assert!(TriangleMesh::new(vec![Vec3::ZERO, Vec3::X], vec![[0, 1, 1]]).is_none());
    assert!(TriangleMesh::new(vec![Vec3::ZERO, Vec3::X], vec![[0, 1, 2]]).is_none());
}

#[test]
fn cuboid_settles_on_mesh_across_multiple_triangles() {
    let mesh = Arc::new(TriangleMesh::heightfield([6, 6], [0.5, 0.5], &[0.0; 36]).unwrap());
    let mut world = PhysicsWorld::new();
    world.add_body(
        RigidBodyBuilder::static_body()
            .shape(Shape::Mesh(mesh))
            .build(),
    );
    let id = world.add_body(
        RigidBodyBuilder::dynamic()
            .position(Vec3::new(1.25, 2.0, 1.25))
            .shape(Shape::cuboid(Vec3::splat(0.4)))
            .build(),
    );
    for _ in 0..240 {
        world.step_fixed().unwrap();
    }
    let body = world.body(id).unwrap();
    assert!(
        body.position().distance(Vec3::new(1.25, 0.4, 1.25)) < 0.01,
        "{body:?}"
    );
    assert!((body.rotation() * Vec3::Y).distance(Vec3::Y) < 0.01);
    assert!(body.is_sleeping());
}

#[test]
fn capsule_side_has_expected_mesh_contact() {
    let mesh = Arc::new(TriangleMesh::heightfield([4, 4], [1.0, 1.0], &[0.0; 16]).unwrap());
    let terrain = RigidBodyBuilder::static_body()
        .shape(Shape::Mesh(mesh))
        .build();
    let capsule = RigidBodyBuilder::dynamic()
        .shape(Shape::capsule(0.4, 0.8))
        .rotation(Quat::from_rotation_z(std::f64::consts::FRAC_PI_2))
        .position(Vec3::new(1.5, 0.35, 1.5))
        .build();
    let mut contacts = Vec::new();
    rust_physics::collision::contacts(&terrain, &capsule, &mut contacts);
    assert_eq!(contacts.len(), 1);
    assert!(contacts[0].normal.distance(Vec3::Y) < 1e-12);
    assert!((contacts[0].depth - 0.05).abs() < 1e-12);
}

#[test]
fn flat_cylinder_manifold_against_terrain_has_expected_depth() {
    let mesh = Arc::new(TriangleMesh::heightfield([3, 3], [2.0, 2.0], &[0.0; 9]).unwrap());
    let terrain = RigidBodyBuilder::static_body()
        .shape(Shape::Mesh(mesh))
        .build();
    let cylinder = RigidBodyBuilder::dynamic()
        .shape(Shape::cylinder(0.35, 0.5))
        .position(Vec3::new(1.0, 0.45, 1.0))
        .build();
    let mut contacts = Vec::new();
    rust_physics::collision::contacts(&terrain, &cylinder, &mut contacts);
    assert!(contacts.len() >= 3, "{contacts:?}");
    for contact in contacts {
        assert!(contact.normal.distance(Vec3::Y) < 1e-5);
        assert!((contact.depth - 0.05).abs() < 1e-5);
    }
}

#[test]
fn convex_capsule_cast_hits_static_heightfield() {
    let mut world = PhysicsWorld::new();
    let mesh = Arc::new(TriangleMesh::heightfield([2, 2], [5.0, 5.0], &[0.0; 4]).unwrap());
    let id = world.add_body(
        RigidBodyBuilder::static_body()
            .shape(Shape::Mesh(mesh))
            .build(),
    );
    let moving = RigidBodyBuilder::kinematic()
        .shape(Shape::capsule(0.3, 0.45))
        .position(Vec3::new(2.0, 3.0, 2.0))
        .build();
    let hit = world
        .cast_shape(&moving, -Vec3::Y, 4.0, QueryFilter::default())
        .unwrap();
    assert_eq!(hit.body, id);
    assert!((hit.distance - 2.25).abs() < 2e-6, "{hit:?}");
    assert!(hit.normal.distance(Vec3::Y) < 1e-5);
}
