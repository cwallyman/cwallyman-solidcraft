//! Reconstruct closed mesh shells as planar B-rep bodies.
use std::collections::{BTreeMap, BTreeSet};

use crate::{Body, KernelError, Result};

fn invalid(reason: &str) -> KernelError {
    KernelError::Invalid(format!("faceted mesh conversion: {reason}"))
}

/// Rebuild each closed shell separately. Nested shells are refused because they may be voids.
pub fn convert_faceted_mesh(body: &Body, tolerance: f64) -> Result<Vec<Body>> {
    crate::guard("convert faceted mesh", || {
        if !tolerance.is_finite() || !(1e-6..=1.0).contains(&tolerance) {
            return Err(invalid("tolerance must be between 0.000001 and 1 mm"));
        }
        let input = body.mesh.as_ref().ok_or_else(|| invalid("select a mesh body"))?;
        if input.triangles.len() > 5_000 {
            return Err(invalid("mesh exceeds the 5000-triangle limit"));
        }
        let welded = crate::mesh_convert::weld(input, tolerance)?;
        let mesh = welded.mesh.as_ref().ok_or_else(|| invalid("empty mesh"))?;
        let mut seen = BTreeSet::new();
        let triangles: Vec<_> = mesh
            .triangles
            .iter()
            .copied()
            .filter(|t| {
                let mut key = *t;
                key.sort_unstable();
                seen.insert(key)
            })
            .collect();
        let mut edges: BTreeMap<(u32, u32), Vec<usize>> = BTreeMap::new();
        for (i, t) in triangles.iter().enumerate() {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                edges.entry((a.min(b), a.max(b))).or_default().push(i);
            }
        }
        let mut neighbours = vec![Vec::new(); triangles.len()];
        for incident in edges.values() {
            if let [a, b] = incident.as_slice() {
                if let Some(n) = neighbours.get_mut(*a) {
                    n.push(*b);
                }
                if let Some(n) = neighbours.get_mut(*b) {
                    n.push(*a);
                }
            }
        }
        let mut visited = vec![false; triangles.len()];
        let mut shells = Vec::new();
        let mut bounds = Vec::new();
        for start in 0..triangles.len() {
            if visited.get(start).copied().unwrap_or(true) {
                continue;
            }
            if shells.len() >= 256 {
                return Err(invalid("mesh exceeds the 256-shell limit"));
            }
            let mut pending = vec![start];
            let mut group = Vec::new();
            while let Some(i) = pending.pop() {
                let Some(mark) = visited.get_mut(i) else {
                    return Err(invalid("invalid triangle index"));
                };
                if *mark {
                    continue;
                }
                *mark = true;
                let Some(&triangle) = triangles.get(i) else {
                    return Err(invalid("invalid triangle index"));
                };
                group.push(triangle);
                pending.extend(neighbours.get(i).into_iter().flatten().copied());
            }
            let mut counts = BTreeMap::<(u32, u32), usize>::new();
            let mut bbox = solidcraft_geom::Aabb3::EMPTY;
            for t in &group {
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    *counts.entry((a.min(b), a.max(b))).or_default() += 1;
                    let point = mesh.positions.get(a as usize).ok_or_else(|| invalid("invalid vertex index"))?;
                    bbox.add(*point);
                }
            }
            if counts.values().any(|&n| n != 2) {
                return Err(invalid("each shell must be closed and manifold"));
            }
            let contains = |a: solidcraft_geom::Aabb3, b: solidcraft_geom::Aabb3| {
                a.min.x <= b.min.x && a.min.y <= b.min.y && a.min.z <= b.min.z && a.max.x >= b.max.x && a.max.y >= b.max.y && a.max.z >= b.max.z
            };
            if bounds.iter().any(|&other| contains(other, bbox) || contains(bbox, other)) {
                return Err(invalid("nested shells may represent cavities and are not supported"));
            }
            bounds.push(bbox);
            let shell = crate::mesh_body(&mesh.positions, &group)?;
            let shell = shell.mesh.as_ref().ok_or_else(|| invalid("empty shell"))?;
            if !crate::meshbody::is_closed(shell) {
                return Err(invalid("shell cannot be consistently oriented"));
            }
            let body = crate::polybool::rebuild_mesh_shell(shell, tolerance)?;
            if !body.validity().is_empty() {
                return Err(invalid("reconstructed shell failed validation"));
            }
            shells.push(body);
        }
        if shells.is_empty() {
            return Err(invalid("empty mesh"));
        }
        Ok(shells)
    })
}
