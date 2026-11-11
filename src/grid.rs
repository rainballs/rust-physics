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
}
