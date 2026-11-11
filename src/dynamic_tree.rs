//! Persistent fat-AABB tree for the rigid-world broad phase.
use crate::Vec3;
use crate::spatial::{Aabb, Proxy};

#[derive(Clone, Copy)]
enum Kind {
    Leaf(usize),
    Branch([usize; 2]),
}

#[derive(Clone, Copy)]
struct Node {
    bounds: Aabb,
    parent: Option<usize>,
    height: usize,
    kind: Kind,
}

#[derive(Default)]
pub(crate) struct DynamicTree {
    nodes: Vec<Option<Node>>,
    free: Vec<usize>,
    leaves: Vec<Option<usize>>,
    current: Vec<Option<Aabb>>,
    next_bounds: Vec<Option<Aabb>>,
    root: Option<usize>,
}

impl DynamicTree {
    pub fn pairs(&mut self, count: usize, proxies: &[Proxy], mut emit: impl FnMut(usize, usize)) {
        self.next_bounds.clear();
        self.next_bounds.resize(count, None);
        for proxy in proxies {
            self.next_bounds[proxy.index] = Some(proxy.bounds);
        }
        for i in 0..self.leaves.len() {
            if self.next_bounds.get(i).is_none_or(Option::is_none)
                && let Some(leaf) = self.leaves[i].take()
            {
                self.detach(leaf);
                self.release(leaf);
            }
        }
        self.leaves.resize(count, None);
        if self.root.is_none() && !proxies.is_empty() {
            self.root = Some(self.build_balanced(&mut proxies.to_vec(), None));
        } else {
            for &Proxy { index, bounds } in proxies {
                let movement = self
                    .current
                    .get(index)
                    .copied()
                    .flatten()
                    .map_or(Vec3::ZERO, |old| {
                        (bounds.min - old.min) * 0.5 + (bounds.max - old.max) * 0.5
                    });
                match self.leaves[index] {
                    Some(leaf)
                        if contains(self.node(leaf).bounds, bounds)
                            && self.node(leaf).bounds.surface_area()
                                <= bounds.surface_area() * 6.0 => {}
                    Some(leaf) => {
                        self.detach(leaf);
                        self.node_mut(leaf).bounds = fatten(bounds, movement);
                        self.insert(leaf);
                    }
                    None => {
                        let leaf = self.alloc(Node {
                            bounds: fatten(bounds, movement),
                            parent: None,
                            height: 0,
                            kind: Kind::Leaf(index),
                        });
                        self.leaves[index] = Some(leaf);
                        self.insert(leaf);
                    }
                }
            }
        }
        std::mem::swap(&mut self.current, &mut self.next_bounds);
        if let Some(root) = self.root {
            self.visit_pairs(root, root, &mut emit);
        }
    }

    fn build_balanced(&mut self, proxies: &mut [Proxy], parent: Option<usize>) -> usize {
        let first = proxies[0];
        let bounds = proxies[1..]
            .iter()
            .fold(first.bounds, |a, b| a.union(b.bounds));
        let i = self.alloc(Node {
            bounds: fatten(bounds, Vec3::ZERO),
            parent,
            height: 0,
            kind: Kind::Leaf(first.index),
        });
        if proxies.len() == 1 {
            self.leaves[first.index] = Some(i);
            return i;
        }
        let extent = bounds.max - bounds.min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };
        let mid = proxies.len() / 2;
        proxies.select_nth_unstable_by(mid, |a, b| {
            a.bounds.center()[axis]
                .total_cmp(&b.bounds.center()[axis])
                .then(a.index.cmp(&b.index))
        });
        let (a, b) = proxies.split_at_mut(mid);
        let left = self.build_balanced(a, Some(i));
        let right = self.build_balanced(b, Some(i));
        self.set_children(i, left, right);
        i
    }

    fn node(&self, i: usize) -> Node {
        self.nodes[i].expect("live node")
    }

    fn node_mut(&mut self, i: usize) -> &mut Node {
        self.nodes[i].as_mut().expect("live node")
    }

    fn alloc(&mut self, node: Node) -> usize {
        if let Some(i) = self.free.pop() {
            self.nodes[i] = Some(node);
            i
        } else {
            let i = self.nodes.len();
            self.nodes.push(Some(node));
            i
        }
    }

    fn release(&mut self, i: usize) {
        self.nodes[i] = None;
        self.free.push(i);
    }

    fn replace(&mut self, parent: Option<usize>, old: usize, new: usize) {
        if let Some(parent) = parent {
            let Kind::Branch(children) = &mut self.node_mut(parent).kind else {
                unreachable!()
            };
            let index = usize::from(children[1] == old);
            debug_assert_eq!(children[index], old);
            children[index] = new;
        } else {
            self.root = Some(new);
        }
        self.node_mut(new).parent = parent;
    }

    fn detach(&mut self, leaf: usize) {
        let Some(parent) = self.node(leaf).parent else {
            self.root = None;
            return;
        };
        let Kind::Branch(children) = self.node(parent).kind else {
            unreachable!()
        };
        let sibling = children[usize::from(children[0] == leaf)];
        let grandparent = self.node(parent).parent;
        self.replace(grandparent, parent, sibling);
        self.node_mut(leaf).parent = None;
        self.release(parent);
        self.refit(grandparent);
    }

    fn insert(&mut self, leaf: usize) {
        let Some(mut sibling) = self.root else {
            self.root = Some(leaf);
            return;
        };
        let bounds = self.node(leaf).bounds;
        loop {
            let node = self.node(sibling);
            let Kind::Branch(children) = node.kind else {
                break;
            };
            let combined = node.bounds.union(bounds).surface_area();
            let inheritance = 2.0 * (combined - node.bounds.surface_area());
            let cost = children.map(|i| {
                let child = self.node(i);
                let total = child.bounds.union(bounds).surface_area();
                inheritance
                    + if child.height == 0 {
                        total
                    } else {
                        total - child.bounds.surface_area()
                    }
            });
            if 2.0 * combined <= cost[0].min(cost[1]) {
                break;
            }
            sibling = children[usize::from(cost[1] < cost[0])];
        }
        let other = self.node(sibling);
        let parent = self.alloc(Node {
            bounds: other.bounds.union(bounds),
            parent: other.parent,
            height: other.height + 1,
            kind: Kind::Branch([sibling, leaf]),
        });
        self.replace(other.parent, sibling, parent);
        self.node_mut(sibling).parent = Some(parent);
        self.node_mut(leaf).parent = Some(parent);
        self.refit(Some(parent));
    }

    fn refit(&mut self, mut current: Option<usize>) {
        while let Some(i) = current {
            self.refresh(i);
            let root = self.balance(i);
            current = self.node(root).parent;
        }
    }

    fn refresh(&mut self, i: usize) {
        if let Kind::Branch([a, b]) = self.node(i).kind {
            let [a, b] = [self.node(a), self.node(b)];
            let node = self.node_mut(i);
            node.bounds = a.bounds.union(b.bounds);
            node.height = 1 + a.height.max(b.height);
        }
    }

    fn set_children(&mut self, i: usize, a: usize, b: usize) {
        self.node_mut(i).kind = Kind::Branch([a, b]);
        self.node_mut(a).parent = Some(i);
        self.node_mut(b).parent = Some(i);
        self.refresh(i);
    }

    fn balance(&mut self, i: usize) -> usize {
        let node = self.node(i);
        let Kind::Branch([a, b]) = node.kind else {
            return i;
        };
        let [ha, hb] = [self.node(a).height, self.node(b).height];
        if ha.abs_diff(hb) <= 1 {
            return i;
        }
        let [small, heavy] = if ha < hb { [a, b] } else { [b, a] };
        let Kind::Branch([u, v]) = self.node(heavy).kind else {
            unreachable!()
        };
        let [moved, retained] = if self.node(u).height <= self.node(v).height {
            [u, v]
        } else {
            [v, u]
        };
        self.replace(node.parent, i, heavy);
        self.set_children(i, small, moved);
        self.set_children(heavy, i, retained);
        heavy
    }

    fn visit_pairs(&self, a: usize, b: usize, emit: &mut impl FnMut(usize, usize)) {
        let [first, second] = [self.node(a), self.node(b)];
        if a == b {
            if let Kind::Branch([l, r]) = first.kind {
                self.visit_pairs(l, l, emit);
                self.visit_pairs(r, r, emit);
                self.visit_pairs(l, r, emit);
            }
            return;
        }
        if !first.bounds.intersects(second.bounds) {
            return;
        }
        match (first.kind, second.kind) {
            (Kind::Leaf(i), Kind::Leaf(j)) => {
                if let (Some(a), Some(b)) = (self.current[i], self.current[j])
                    && a.intersects(b)
                {
                    emit(i, j);
                }
            }
            (Kind::Branch([l, r]), _) if first.height >= second.height => {
                self.visit_pairs(l, b, emit);
                self.visit_pairs(r, b, emit);
            }
            (_, Kind::Branch([l, r])) => {
                self.visit_pairs(a, l, emit);
                self.visit_pairs(a, r, emit);
            }
            _ => unreachable!("distinct leaves or branch"),
        }
    }
}

fn fatten(bounds: Aabb, movement: Vec3) -> Aabb {
    let extent = bounds.max - bounds.min;
    let margin = (extent * 0.02).max(Vec3::splat(0.001));
    let shift = (movement * 2.0).clamp(-extent, extent);
    Aabb {
        min: bounds.min - margin + shift.min(Vec3::ZERO),
        max: bounds.max + margin + shift.max(Vec3::ZERO),
    }
}

fn contains(outer: Aabb, inner: Aabb) -> bool {
    outer.min.cmple(inner.min).all() && inner.max.cmple(outer.max).all()
}
