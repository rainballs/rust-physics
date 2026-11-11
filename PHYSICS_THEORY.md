# Physics theory

## Rigid motion

Positions and velocities use double-precision Cartesian coordinates. Example scenes use metres, seconds, kilograms, and +Y up. The rigid world uses semi-implicit Euler. Update velocity from acceleration and torque first, then advance the pose. Exponential damping multiplies linear velocity and angular momentum by exp(−d Δt).

Dynamic bodies store world angular momentum L. Their angular velocity is I_world⁻¹ L, where I_world⁻¹ = R I_local⁻¹ Rᵀ. The pose integrator estimates angular velocity at the rotational midpoint and applies its quaternion exponential to the original orientation. Torque-free L remains constant, while rotational energy depends on integration accuracy. The sphere, solid cuboid, and uniform solid capsule use analytical principal inertia. The capsule combines a cylinder with two hemispheres using the parallel-axis theorem. Cylinders use the uniform solid-cylinder inertia tensor.

Standalone integrators implement explicit and semi-implicit Euler, velocity Verlet, and classical RK4 for acceleration that depends only on position. Euler variants are first order, velocity Verlet is second order, and RK4 is fourth order for smooth fields at a fixed finite step. Symplectic methods are generally preferable for long, undamped orbital motion at a well-resolved fixed timestep. Adaptive RK4 uses step doubling, Richardson extrapolation, and separate position and velocity error scales. Its tolerance controls local error.

## Contact geometry and response

Cuboids test three face normals from each body plus up to nine non-degenerate edge cross products. A face-axis contact clips an incident quadrilateral against the four side planes of the reference face. An edge-axis contact uses the closest finite segment pair. Capsules reduce to segment closest points or a piecewise quadratic segment-to-box calculation. Planes are solid half-spaces with outward local +Y. Experimental GJK finds the closest point on a simplex of the Minkowski difference. EPA expands an enclosing tetrahedron toward the nearest supporting boundary and recovers witnesses. Iteration budgets and shallow, nearly degenerate intersections can prevent a usable manifold.

For unit contact normal n, offsets rᵢ from each center, and inverse masses wᵢ, the denominator for a normal impulse is the sum of wᵢ + (rᵢ × n) · Iᵢ⁻¹(rᵢ × n). The impulse is accumulated and projected onto nonnegative values. Restitution sets a separating target speed for sufficiently fast impacts. Tangential impulses first test a static Coulomb bound and use a smaller dynamic bound on slip. Position correction translates each body in proportion to its inverse mass. The default penetration slop is 0.001 world units.

CCD computes an entering time of impact for a sphere moving linearly over one substep. A box expanded by a sphere has rounded edges and corners. Between crossings of its unexpanded coordinate planes, distance to the closest box feature is quadratic, which gives the entering impact time without replacing the rounded corners by square ones. Cylinder casts instead use conservative advancement against the signed cylinder surface distance.

## Constraints

Bilateral joint rows target a relative velocity with a bounded positional-error bias. Rope and limit rows are unilateral. Within their slack they allow enough speed to consume the remaining slack over one substep. Motor impulses are capped by the force or torque limit times Δt. An implicit damped spring adds a softness term 1/[Δt(c + Δt k)] for damping c and stiffness k.

XPBD uses accumulated multipliers over the iterations of each step. For constraint value C, gradients ∇Cᵢ, inverse masses wᵢ, and scaled compliance α = compliance/Δt², the multiplier update is (−C − αλ)/(Σ wᵢ|∇Cᵢ|² + α). Position updates use wᵢ∇Cᵢ times that multiplier increment. Cloth bends via longer distance links. Tetrahedral volume is a signed scalar triple product divided by six.

## Experimental SPH

SPH density is a weighted sum using the poly6 kernel. The pressure model uses a linear equation of state and clamps negative pressure. Forces use the spiky gradient and viscosity Laplacian. Pair accelerations are equal and opposite for identical particle masses. The spatial hash traverses 27 neighboring cells of width equal to the smoothing radius. Boundary projection gives a simple one-way collision model, and the acoustic timestep estimate is a starting point for resolving pressure propagation.
