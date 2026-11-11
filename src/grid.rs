//! Deterministic neighbor iteration over a uniform spatial hash.
use std::collections::HashMap;

use crate::Vec3;

/// Sparse cells with a fixed width. Each cell preserves insertion order, and
/// queries traverse cells in coordinate order.
#[derive(Clone, Debug)]
pub struct SpatialHash {
    width: f64,
    cells: HashMap<[i64; 3], Vec<usize>>,
    recycled: Vec<Vec<usize>>,
}

impl SpatialHash {
    pub fn new(width: f64) -> Option<Self> {
        (width.is_finite() && width > 0.0).then(|| Self {
            width,
            cells: HashMap::new(),
            recycled: Vec::new(),
        })
    }

    fn cell(&self, point: Vec3) -> [i64; 3] {
        (point / self.width).to_array().map(|v| v.floor() as i64)
    }

    pub fn rebuild(&mut self, positions: &[Vec3]) {
        for (_, mut cell) in self.cells.drain() {
            cell.clear();
            self.recycled.push(cell);
        }
        for (index, &point) in positions.iter().enumerate() {
            if point.is_finite() {
                let key = self.cell(point);
                self.cells
                    .entry(key)
                    .or_insert_with(|| self.recycled.pop().unwrap_or_default())
                    .push(index);
            }
        }
    }

    /// Candidates within one cell of the point's cell. A distance check is
    /// required for exact searches up to the cell width.
    pub fn neighbors(&self, point: Vec3) -> impl Iterator<Item = usize> + '_ {
        let [x, y, z] = self.cell(point);
        let span = |v: i64| v.saturating_sub(1)..=v.saturating_add(1);
        span(x)
            .flat_map(move |i| {
                span(y).flat_map(move |j| span(z).filter_map(move |k| self.cells.get(&[i, j, k])))
            })
            .flatten()
            .copied()
    }
    /// Unique neighbor pairs at distances strictly below one cell width.
    pub(crate) fn pairs<'a>(
        &'a self,
        positions: &'a [Vec3],
    ) -> impl Iterator<Item = (usize, usize)> + 'a {
        let limit = self.width * self.width;
        positions.iter().enumerate().flat_map(move |(i, &p)| {
            self.neighbors(p)
                .filter(move |&j| j > i && p.distance_squared(positions[j]) < limit)
                .map(move |j| (i, j))
        })
    }
}

use crate::spatial::Proxy;

/// Uniform grid over full AABB extents. Very large proxies use a direct
/// fallback to bound cell replication.
#[derive(Default)]
pub(crate) struct ProxyGrid {
    cells: HashMap<[i64; 3], Vec<usize>>,
    recycled: Vec<Vec<usize>>,
    low: Vec<[i64; 3]>,
    large: Vec<usize>,
    large_flags: Vec<bool>,
}

impl ProxyGrid {
    pub fn pairs(&mut self, width: f64, proxies: &[Proxy], mut emit: impl FnMut(usize, usize)) {
        for (_, mut bucket) in self.cells.drain() {
            bucket.clear();
            self.recycled.push(bucket);
        }
        self.low.clear();
        self.large.clear();
        self.large_flags.clear();
        let coord = |p: Vec3| (p / width).to_array().map(|v| v.floor() as i64);
        for (i, proxy) in proxies.iter().enumerate() {
            let low = coord(proxy.bounds.min);
            let high = coord(proxy.bounds.max);
            self.low.push(low);
            let span = std::array::from_fn::<_, 3, _>(|j| {
                high[j].saturating_sub(low[j]).saturating_add(1)
            });
            let large = span.into_iter().fold(1_i64, i64::saturating_mul) > 512;
            self.large_flags.push(large);
            if large {
                self.large.push(i);
                continue;
            }
            for x in low[0]..=high[0] {
                for y in low[1]..=high[1] {
                    for z in low[2]..=high[2] {
                        self.cells
                            .entry([x, y, z])
                            .or_insert_with(|| self.recycled.pop().unwrap_or_default())
                            .push(i);
                    }
                }
            }
        }
        for (&cell, bucket) in &self.cells {
            for (i, &a) in bucket.iter().enumerate() {
                for &b in &bucket[i + 1..] {
                    let first_common =
                        std::array::from_fn(|axis| self.low[a][axis].max(self.low[b][axis]));
                    if cell == first_common && proxies[a].bounds.intersects(proxies[b].bounds) {
                        emit(proxies[a].index, proxies[b].index);
                    }
                }
            }
        }
        for &a in &self.large {
            for (b, proxy) in proxies.iter().enumerate() {
                if (!self.large_flags[b] || a < b) && proxies[a].bounds.intersects(proxy.bounds) {
                    emit(proxies[a].index, proxy.index);
                }
            }
        }
    }
}
