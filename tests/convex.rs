use rust_physics::collision::contacts;
use rust_physics::convex::{ConvexContact, ConvexHull, Transformed, query};
use rust_physics::{Quat, RigidBodyBuilder, Shape, Vec3};

fn cube() -> ConvexHull {
    let vertices = (0..8)
        .map(|i| {
            Vec3::new(
                if i & 1 == 0 { -1.0 } else { 1.0 },
                if i & 2 == 0 { -1.0 } else { 1.0 },
                if i & 4 == 0 { -1.0 } else { 1.0 },
            )
        })
        .collect::<Vec<_>>();
    ConvexHull::new(vertices).unwrap()
}

#[test]
fn axis_aligned_hulls_detect_gap_touch_and_penetration() {
    let shape = cube();
    let second = |x| Transformed {
        shape: &shape,
        position: Vec3::X * x,
        rotation: Quat::IDENTITY,
    };
    assert_eq!(query(&shape, &second(3.0), 1e-8), ConvexContact::Separated);
    assert_eq!(query(&shape, &second(2.0), 1e-8), ConvexContact::Touching);
    let ConvexContact::Penetrating {
        depth,
        normal,
        point_a,
        point_b,
    } = query(&shape, &second(1.75), 1e-8)
    else {
        panic!("expected penetration");
    };
    assert!((depth - 0.25).abs() < 1e-12);
    assert_eq!(normal, Vec3::X);
    assert!((point_a.x - 1.0).abs() < 1e-12);
    assert!((point_b.x - 0.75).abs() < 1e-12);
}

#[test]
fn generated_hull_pairs_agree_with_oriented_box_sat() {
    let shape = cube();
    let first = RigidBodyBuilder::static_body()
        .shape(Shape::cuboid(Vec3::ONE))
        .build();
    let mut found = [0; 2];
    let mut points = Vec::new();
    for i in 0..384 {
        let t = f64::from(i);
        let position = Vec3::new((t * 0.723).sin(), (t * 0.439).cos(), (t * 0.315).sin()) * 2.8;
        let rotation = Quat::from_scaled_axis(Vec3::new(t * 0.013, t * 0.017, t * 0.011));
        let hull = Transformed {
            shape: &shape,
            position,
            rotation,
        };
        let second = RigidBodyBuilder::static_body()
            .shape(Shape::cuboid(Vec3::ONE))
            .position(position)
            .rotation(rotation)
            .build();
        points.clear();
        contacts(&first, &second, &mut points);
        let result = query(&shape, &hull, 1e-8);
        if points.is_empty() {
            assert_eq!(result, ConvexContact::Separated, "{i}");
            found[0] += 1;
        } else {
            let ConvexContact::Penetrating {
                normal,
                depth,
                point_a,
                point_b,
            } = result
            else {
                panic!("{i}: {result:?}");
            };
            assert!(normal.is_normalized() && depth.is_finite() && depth > 0.0);
            assert!((point_a - point_b).distance(normal * depth) < 1e-7, "{i}");
            let moved = Transformed {
                position: position + normal * (depth + 1e-6),
                ..hull
            };
            assert_eq!(query(&shape, &moved, 1e-8), ConvexContact::Separated, "{i}");
            found[1] += 1;
        }
    }
    assert!(found.into_iter().all(|n| n > 50));
}

#[test]
fn planar_hulls_are_rejected() {
    assert!(ConvexHull::new(vec![Vec3::ZERO, Vec3::X, Vec3::Z, Vec3::X + Vec3::Z]).is_none());
}

#[test]
fn cylinder_box_uses_support_map_and_parallel_cylinders_resolve() {
    let cuboid = RigidBodyBuilder::static_body()
        .shape(Shape::cuboid(Vec3::ONE))
        .build();
    let cylinder = |p| {
        RigidBodyBuilder::static_body()
            .shape(Shape::cylinder(0.5, 0.7))
            .position(p)
            .build()
    };
    let mut output = Vec::new();
    contacts(&cuboid, &cylinder(Vec3::X * 1.4), &mut output);
    assert_eq!(output.len(), 1);
    assert!((output[0].depth - 0.1).abs() < 1e-5);
    assert!(output[0].normal.distance(Vec3::X) < 1e-5);
    output.clear();
    contacts(&cuboid, &cylinder(Vec3::X * 1.6), &mut output);
    assert!(output.is_empty());
    contacts(&cylinder(Vec3::ZERO), &cylinder(Vec3::ZERO), &mut output);
    assert_eq!(output.len(), 2);
    assert!(
        output
            .iter()
            .all(|c| (c.depth - 1.0).abs() < 1e-12 && c.normal.is_normalized())
    );
}

#[test]
fn rigid_tetrahedron_centroid_inertia_and_surface_queries() {
    use rust_physics::hull::RigidHull;
    use rust_physics::{PhysicsWorld, QueryFilter};
    use std::sync::Arc;
    let vertices = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
    let faces = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
    let mut reversed = faces.clone();
    reversed[0].swap(1, 2);
    assert!(RigidHull::new(vertices.clone(), reversed).is_none());
    assert!(RigidHull::new(vertices.clone(), faces[..3].to_vec()).is_none());
    let hull = Arc::new(RigidHull::new(vertices, faces).unwrap());
    assert!(hull.input_center().distance(Vec3::splat(0.25)) < 1e-12);
    let mut w = PhysicsWorld::new();
    let id = w.add_body(
        RigidBodyBuilder::dynamic()
            .shape(Shape::Hull(hull))
            .mass(2.0)
            .build(),
    );
    let b = w.body(id).unwrap();
    let tensor = b.world_inverse_inertia().inverse();
    assert!((tensor.x_axis.x - 0.15).abs() < 1e-12);
    assert!((tensor.x_axis.y - 0.025).abs() < 1e-12);
    let hit = w
        .raycast(Vec3::ZERO, -Vec3::X, 1.0, QueryFilter::default())
        .unwrap();
    assert!((hit.distance - 0.25).abs() < 1e-12);
    assert_eq!(hit.normal, -Vec3::X);
    let surface = w
        .nearest_surface(Vec3::ZERO, QueryFilter::default())
        .unwrap();
    assert!(surface.distance < 0.0 && surface.distance > -0.25);
    assert_eq!(
        w.overlap_sphere(Vec3::ZERO, 0.01, QueryFilter::default())
            .collect::<Vec<_>>(),
        [id]
    );
}

#[test]
fn gjk_distance_returns_closest_witnesses() {
    use rust_physics::convex::{ConvexDistance, distance};
    let cube = cube();
    let moved = Transformed {
        shape: &cube,
        position: Vec3::new(3.0, 3.0, 0.0),
        rotation: Quat::IDENTITY,
    };
    let ConvexDistance::Separated {
        distance,
        normal,
        point_a,
        point_b,
    } = distance(&cube, &moved, 1e-9)
    else {
        panic!("expected distance");
    };
    assert!((distance - 2.0_f64.sqrt()).abs() < 1e-12);
    assert!(normal.distance((Vec3::X + Vec3::Y).normalize()) < 1e-12);
    assert!((point_b - point_a).distance(Vec3::X + Vec3::Y) < 1e-12);
}

#[test]
fn point_cloud_hull_removes_interior_points_and_matches_cube_mass_properties() {
    use rust_physics::hull::RigidHull;
    use std::sync::Arc;
    let source = cube();
    let mut points = source.points().to_vec();
    points.extend([Vec3::ZERO, Vec3::X * 0.3, Vec3::ONE, Vec3::Y * 0.7]);
    points.reverse();
    let hull = Arc::new(RigidHull::from_points(points).unwrap());
    assert_eq!(hull.surface().vertices().len(), 8);
    assert_eq!(hull.surface().triangles().len(), 12);
    assert!(hull.input_center().length() < 1e-12);
    let b = RigidBodyBuilder::dynamic()
        .shape(Shape::Hull(hull))
        .mass(3.0)
        .build();
    let i = b.world_inverse_inertia();
    for axis in Vec3::AXES {
        assert!((i * axis).distance(axis * 0.5) < 1e-12);
    }
}

#[test]
fn generated_hull_support_matches_original_point_cloud() {
    use rust_physics::convex::{PrimitiveSupport, SupportMap};
    use rust_physics::hull::RigidHull;
    use std::sync::Arc;
    let points: Vec<_> = (0..180)
        .map(|i| {
            let y = 1.0 - 2.0 * (f64::from(i) + 0.5) / 180.0;
            let r = (1.0 - y * y).sqrt();
            let angle = 2.3999632297 * f64::from(i);
            Vec3::new(1.3 * r * angle.cos(), 0.7 * y, 1.8 * r * angle.sin())
                + Vec3::new(2.0, 3.0, -1.0)
        })
        .collect();
    let source = ConvexHull::new(points.clone()).unwrap();
    let hull = Arc::new(RigidHull::from_points(points).unwrap());
    let body = RigidBodyBuilder::static_body()
        .position(hull.input_center())
        .shape(Shape::Hull(hull))
        .build();
    let generated = PrimitiveSupport::new(&body).unwrap();
    for i in 0..128 {
        let t = f64::from(i);
        let direction = Vec3::new(t.sin(), (0.29 * t).cos(), (0.83 * t).sin()).normalize();
        assert!(
            (source.support(direction).dot(direction)
                - generated.support(direction).dot(direction))
            .abs()
                < 1e-10
        );
    }
}

#[test]
fn hull_face_clipping_generates_a_supporting_contact_patch() {
    use rust_physics::hull::RigidHull;
    use std::sync::Arc;
    let hull = Arc::new(RigidHull::from_points(cube().points().iter().copied()).unwrap());
    let a = RigidBodyBuilder::static_body()
        .shape(Shape::Hull(hull.clone()))
        .build();
    let b = RigidBodyBuilder::dynamic()
        .shape(Shape::Hull(hull))
        .position(Vec3::Y * 1.9)
        .build();
    let mut output = Vec::new();
    contacts(&a, &b, &mut output);
    assert_eq!(output.len(), 4, "{output:?}");
    assert!(
        output
            .iter()
            .all(|p| (p.depth - 0.1).abs() < 1e-8 && p.normal.distance(Vec3::Y) < 1e-8)
    );
}

#[test]
fn aligned_cylinder_caps_produce_contact_points_in_the_disk_lens() {
    let a = RigidBodyBuilder::static_body()
        .shape(Shape::cylinder(0.8, 0.4))
        .build();
    let b = RigidBodyBuilder::dynamic()
        .shape(Shape::cylinder(0.7, 0.4))
        .position(Vec3::new(0.35, 0.75, 0.0))
        .build();
    let mut output = Vec::new();
    contacts(&a, &b, &mut output);
    assert_eq!(output.len(), 4);
    for c in output {
        assert!((c.depth - 0.05).abs() < 1e-12 && c.normal == Vec3::Y);
        let p = Vec3::new(c.point.x, 0.0, c.point.z);
        assert!(p.length() <= 0.8 + 1e-12 && p.distance(Vec3::X * 0.35) <= 0.7 + 1e-12);
    }
}
