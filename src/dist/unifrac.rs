//! UniFrac phylogenetic distance implementations.

use super::traits::Distance;

use anyhow::{anyhow, Result};
use log::debug;
use phylotree::tree::Tree;
use std::collections::{HashMap, HashSet};

// for BitVec used in NewDistUniFrac
use bitvec::prelude::*;
use newick::{one_from_string, NewickTree};
use succparen::{
    bitwise::SparseOneNnd,
    tree::{
        balanced_parens::{BalancedParensTree, Node as BpNode},
        traversal::{DepthFirstTraverse, VisitNode},
        LabelVec, Node,
    },
};

// for DistUniFrac_C FFI
use std::os::raw::{c_char, c_double, c_uint};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// DistUniFrac
#[derive(Default, Clone)]
pub struct DistUniFrac {
    /// Weighted or unweighted
    weighted: bool,
    /// tint[i] = parent of node i
    tint: Vec<usize>,
    /// lint[i] = length of edge from node i up to tint[i]
    lint: Vec<f32>,
    /// Postorder nodes
    nodes_in_order: Vec<usize>,
    /// node_name_map[node_in_order_idx] = Some("T4") or whatever
    #[allow(dead_code)]
    node_name_map: Vec<Option<String>>,
    /// leaf_map: "T4" -> postorder index
    leaf_map: HashMap<String, usize>,
    /// total tree length (not used to normalize in this example)
    #[allow(dead_code)]
    total_tree_length: f32,
    /// Feature names in the same order as va,vb
    feature_names: Vec<String>,
}

impl DistUniFrac {
    /// Build DistUniFrac from the given `newick_str` (no re-root),
    /// a boolean `weighted` for Weighted or Unweighted, plus your feature names.
    ///
    /// *Important*: We do not compress or re-root the tree. This ensures T4's path
    pub fn new(newick_str: &str, weighted: bool, feature_names: Vec<String>) -> Result<Self> {
        // Parse the tree from the Newick string
        let tree = Tree::from_newick(newick_str)?;

        // Build arrays (same approach as your old code)
        let (tint, lint, nodes_in_order, node_name_map) = build_tint_lint(&tree)?;

        let leaf_map = build_leaf_map(&tree, &node_name_map)?;

        let total_tree_length: f32 = lint.iter().sum();

        Ok(Self {
            weighted,
            tint,
            lint,
            nodes_in_order,
            node_name_map,
            leaf_map,
            total_tree_length,
            feature_names,
        })
    }
}

/// Implement the Distance<f32> trait
impl Distance<f32> for DistUniFrac {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        debug!(
            "DistUniFrac eval called: weighted={}, #features={}",
            self.weighted,
            self.feature_names.len()
        );
        if self.weighted {
            compute_unifrac_for_pair_weighted_bitwise(
                &self.tint,
                &self.lint,
                &self.nodes_in_order,
                &self.leaf_map,
                &self.feature_names,
                va,
                vb,
            )
        } else {
            compute_unifrac_for_pair_unweighted_bitwise(
                &self.tint,
                &self.lint,
                &self.nodes_in_order,
                &self.leaf_map,
                &self.feature_names,
                va,
                vb,
            )
        }
    }
}

//--------------------------------------------------------------------------------------//
//--------------------------------------------------------------------------------------//
fn build_tint_lint(tree: &Tree) -> Result<(Vec<usize>, Vec<f32>, Vec<usize>, Vec<Option<String>>)> {
    let root = tree.get_root().map_err(|_| anyhow!("Tree has no root"))?;
    let postord = tree.postorder(&root)?;
    debug!("postord = {:?}", postord);
    let num_nodes = postord.len();

    // node_id -> postorder index
    let mut pos_map = vec![0; tree.size()];
    for (i, &nid) in postord.iter().enumerate() {
        pos_map[nid] = i;
    }

    let mut tint = vec![0; num_nodes];
    let mut lint = vec![0.0; num_nodes];
    let mut node_name_map = vec![None; num_nodes];

    // Mark the root as its own ancestor
    let root_idx = pos_map[root];
    tint[root_idx] = root_idx;
    lint[root_idx] = 0.0;

    // Fill in parent/edge arrays
    for &nid in &postord {
        let node = tree.get(&nid)?;
        if let Some(name) = &node.name {
            node_name_map[pos_map[nid]] = Some(name.clone());
        }
        if nid != root {
            let p = node
                .parent
                .ok_or_else(|| anyhow!("Node has no parent but is not root"))?;
            tint[pos_map[nid]] = pos_map[p];
            lint[pos_map[nid]] = node.parent_edge.unwrap_or(0.0) as f32;
        }
    }
    Ok((tint, lint, postord, node_name_map))
}

fn build_leaf_map(
    tree: &Tree,
    _node_name_map: &[Option<String>],
) -> Result<HashMap<String, usize>> {
    let root = tree.get_root().map_err(|_| anyhow!("Tree has no root"))?;
    let postord = tree.postorder(&root)?;
    debug!("postord = {:?}", postord);
    let mut pos_map = vec![0; tree.size()];
    for (i, &nid) in postord.iter().enumerate() {
        pos_map[nid] = i;
    }

    let mut leaf_map = HashMap::new();
    for l in tree.get_leaves() {
        let node = tree.get(&l)?;
        if node.is_tip() {
            if let Some(name) = &node.name {
                let idx = pos_map[l];
                debug!("   => recognized tip='{}', postord_idx={}", name, idx);
                leaf_map.insert(name.clone(), idx);
            }
        }
    }
    Ok(leaf_map)
}

// Helper function to extract leaf names from newick string without phylotree dependency
#[allow(dead_code)]
fn extract_leaf_names_from_newick_string(newick_str: &str) -> Result<Vec<String>> {
    let mut leaf_names = Vec::new();

    // Method 1: Try to extract from existing tree iteration
    let t: NewickTree =
        one_from_string(newick_str).map_err(|e| anyhow!("Failed to parse Newick string: {}", e))?;

    // Create a temporary SuccTrav to iterate through nodes
    let mut temp_lens = Vec::<f32>::new();
    let trav = SuccTrav::new(&t, &mut temp_lens);
    let bp: BalancedParensTree<LabelVec<usize>, SparseOneNnd> =
        BalancedParensTree::new_builder(trav, LabelVec::<usize>::new()).build_all();

    let total = bp.len() + 1;
    let mut kids = vec![Vec::<usize>::new(); total];
    let mut post = Vec::<usize>::with_capacity(total);
    collect_children::<SparseOneNnd>(&bp.root(), &mut kids, &mut post);

    // Find leaf nodes (nodes with no children)
    let mut leaf_ids = Vec::<usize>::new();
    for node_id in 0..total {
        if kids[node_id].is_empty() {
            leaf_ids.push(node_id);
        }
    }

    // Method 2: Use simple pattern matching for common cases
    // This handles most newick formats including complex ones
    let cleaned = newick_str
        .replace('\n', "")
        .replace('\r', "")
        .replace('\t', " ");

    // Find leaf patterns: word followed by : (leaf with branch length) or word at end before )
    let mut current_token = String::new();
    let mut chars = cleaned.chars().peekable();
    let mut depth = 0;

    while let Some(ch) = chars.next() {
        match ch {
            '(' => {
                depth += 1;
                if !current_token.is_empty() && depth == 1 {
                    // Root name, skip
                    current_token.clear();
                }
            }
            ')' => {
                depth -= 1;
                // Check if we have accumulated a token before closing
                if !current_token.trim().is_empty() {
                    let name = current_token.trim().to_string();
                    if !name.is_empty() && !name.chars().all(|c| c.is_numeric() || c == '.') {
                        leaf_names.push(name);
                    }
                }
                current_token.clear();
            }
            ',' => {
                // End of current item
                if !current_token.trim().is_empty() {
                    let name = current_token.trim().to_string();
                    if !name.is_empty() && !name.chars().all(|c| c.is_numeric() || c == '.') {
                        leaf_names.push(name);
                    }
                }
                current_token.clear();
            }
            ':' => {
                // Branch length follows, current token might be a leaf name
                if !current_token.trim().is_empty() {
                    let name = current_token.trim().to_string();
                    if !name.is_empty() && !name.chars().all(|c| c.is_numeric() || c == '.') {
                        leaf_names.push(name);
                    }
                }
                current_token.clear();

                // Skip until next delimiter
                while let Some(&next_ch) = chars.peek() {
                    if next_ch == ',' || next_ch == ')' || next_ch == '(' {
                        break;
                    }
                    chars.next();
                }
            }
            ';' => break,
            ' ' => {
                // Keep spaces in names but trim later
                if !current_token.is_empty() {
                    current_token.push(ch);
                }
            }
            _ => {
                current_token.push(ch);
            }
        }
    }

    // Remove duplicates and empty names
    leaf_names.sort();
    leaf_names.dedup();
    leaf_names.retain(|name| !name.is_empty());

    Ok(leaf_names)
}

// start of NewDistUniFrac

/// A new DistUniFrac struct using succparen and balanced parentheses tree.
#[derive(Debug)]
pub struct NewDistUniFrac {
    pub weighted: bool,
    pub post: Vec<usize>,
    pub kids: Vec<Vec<usize>>,
    pub lens: Vec<f32>,
    pub leaf_ids: Vec<usize>,
    pub feature_names: Vec<String>,
    /// Parent mapping: parent[i] = parent node index of node i, or i if root
    pub parent: Vec<usize>,
    /// Index of root node (last node in postorder)
    pub root_idx: usize,
}

impl NewDistUniFrac {
    /// Build NewDistUniFrac using succparen approach with corrected tree construction
    pub fn new(newick_str: &str, weighted: bool, feature_names: Vec<String>) -> Result<Self> {
        // 1. Parse Newick string
        let t: NewickTree = one_from_string(newick_str)
            .map_err(|e| anyhow!("Failed to parse Newick string: {}", e))?;

        // 2. Build tree structure using succparen-compatible approach
        // Map Newick nodes to sequential indices (postorder) for succparen compatibility
        let mut newick_to_index = HashMap::new();
        let mut index_counter = 0;
        let mut lens = Vec::<f32>::new();

        // Assign indices in postorder (required for succparen/unifrac_bp pattern)
        fn assign_postorder_indices(
            node_id: usize,
            tree: &NewickTree,
            mapping: &mut HashMap<usize, usize>,
            counter: &mut usize,
            lens: &mut Vec<f32>,
        ) {
            // Process children first (postorder)
            for &child_id in tree[node_id].children() {
                assign_postorder_indices(child_id, tree, mapping, counter, lens);
            }

            // Assign index to current node
            mapping.insert(node_id, *counter);

            // Store branch length
            if *counter >= lens.len() {
                lens.resize(*counter + 1, 0.0);
            }
            lens[*counter] = tree[node_id].branch().copied().unwrap_or(0.0);

            *counter += 1;
        }

        assign_postorder_indices(
            t.root(),
            &t,
            &mut newick_to_index,
            &mut index_counter,
            &mut lens,
        );

        // 3. Build children arrays and postorder (succparen-compatible structure)
        let total_nodes = index_counter;
        let mut kids = vec![Vec::<usize>::new(); total_nodes];
        let mut post = Vec::<usize>::with_capacity(total_nodes);

        fn build_succparen_structure(
            node_id: usize,
            tree: &NewickTree,
            mapping: &HashMap<usize, usize>,
            kids: &mut [Vec<usize>],
            post: &mut Vec<usize>,
        ) {
            let current_idx = mapping[&node_id];

            // Process children in correct order
            for &child_id in tree[node_id].children() {
                let child_idx = mapping[&child_id];
                kids[current_idx].push(child_idx);
                build_succparen_structure(child_id, tree, mapping, kids, post);
            }
            post.push(current_idx);
        }

        build_succparen_structure(t.root(), &t, &newick_to_index, &mut kids, &mut post);

        // 3.5. Build parent mapping from kids array
        // Root is the last node in postorder (processed last in postorder traversal)
        let root_idx = post[post.len() - 1];
        let mut parent = vec![0; total_nodes];
        
        // Build parent mapping: for each node, set parent of its children
        for (node_idx, children) in kids.iter().enumerate() {
            for &child_idx in children {
                if child_idx < total_nodes {
                    parent[child_idx] = node_idx;
                }
            }
        }
        
        // Root is its own parent
        parent[root_idx] = root_idx;

        // 4. Find leaf nodes using original Newick tree (correct approach)
        let mut leaf_ids = Vec::<usize>::new();
        let mut leaf_nm = Vec::<String>::new();

        fn find_leaves_recursive(
            node_id: usize,
            tree: &NewickTree,
            mapping: &HashMap<usize, usize>,
            leaf_ids: &mut Vec<usize>,
            leaf_nm: &mut Vec<String>,
        ) {
            let node = &tree[node_id];

            if node.children().is_empty() {
                if let Some(name) = node.data().name.as_ref() {
                    if let Some(&idx) = mapping.get(&node_id) {
                        leaf_ids.push(idx);
                        leaf_nm.push(name.clone());
                    }
                }
            } else {
                for &child_id in node.children() {
                    find_leaves_recursive(child_id, tree, mapping, leaf_ids, leaf_nm);
                }
            }
        }

        find_leaves_recursive(t.root(), &t, &newick_to_index, &mut leaf_ids, &mut leaf_nm);

        // 5. Create taxon-name ? leaf-index mapping
        let t2leaf: HashMap<&str, usize> = leaf_nm
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();

        // 6. Map feature_names to leaf_ids
        let mut mapped_leaf_ids = Vec::with_capacity(feature_names.len());
        for fname in &feature_names {
            if let Some(&leaf_pos) = t2leaf.get(fname.as_str()) {
                mapped_leaf_ids.push(leaf_ids[leaf_pos]);
            } else {
                return Err(anyhow!("Feature name '{}' not found in tree", fname));
            }
        }

        Ok(Self {
            weighted,
            post,
            kids,
            lens,
            leaf_ids: mapped_leaf_ids,
            feature_names,
            parent,
            root_idx,
        })
    }

    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    pub fn num_features(&self) -> usize {
        self.feature_names.len()
    }
}

impl Distance<f32> for NewDistUniFrac {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        if self.weighted {
            unifrac_pair_weighted(&self.post, &self.kids, &self.lens, &self.leaf_ids, va, vb) as f32
        } else {
            // Convert to bit vectors for the working unifrac_pair function
            let a_bits: BitVec<u8, bitvec::order::Lsb0> = va.iter().map(|&x| x > 0.0).collect();
            let b_bits: BitVec<u8, bitvec::order::Lsb0> = vb.iter().map(|&x| x > 0.0).collect();

            // Use the proven working unifrac_pair function
            unifrac_pair(
                &self.post,
                &self.kids,
                &self.lens,
                &self.leaf_ids,
                &a_bits,
                &b_bits,
            ) as f32
        }
    }
}

/// UniFrac using succparen tree structure with shared/union logic (like unifrac_pair)
#[allow(dead_code)]
fn unifrac_succparen_normalized(
    post: &[usize],
    kids: &[Vec<usize>],
    lens: &[f32],
    leaf_ids: &[usize],
    va: &[f32],
    vb: &[f32],
) -> f64 {
    const A_BIT: u8 = 0b01;
    const B_BIT: u8 = 0b10;

    // Create presence masks (like unifrac_pair)
    let mut mask = vec![0u8; lens.len()];

    println!("DEBUG: Setting leaf masks:");
    for (leaf_pos, &leaf_node_id) in leaf_ids.iter().enumerate() {
        if leaf_pos < va.len() && leaf_pos < vb.len() {
            let a_val = va[leaf_pos];
            let b_val = vb[leaf_pos];
            println!(
                "DEBUG: Leaf {} (BP node {}): A={}, B={}",
                leaf_pos, leaf_node_id, a_val, b_val
            );

            if va[leaf_pos] > 0.0 {
                mask[leaf_node_id] |= A_BIT;
            }
            if vb[leaf_pos] > 0.0 {
                mask[leaf_node_id] |= B_BIT;
            }

            println!(
                "DEBUG: Mask for BP node {} = {:02b}",
                leaf_node_id, mask[leaf_node_id]
            );
        }
    }

    // Propagate masks up the tree (like unifrac_pair), but only for internal nodes
    println!("DEBUG: Propagating masks up the tree:");
    for &node in post {
        // Only propagate for nodes that have children (internal nodes)
        if kids[node].is_empty() {
            continue; // Skip leaf nodes in the tree structure
        }

        let initial_mask = mask[node];
        for &child in &kids[node] {
            println!(
                "DEBUG: Node {} gets mask {:02b} from child {}",
                node, mask[child], child
            );
            mask[node] |= mask[child];
        }
        if initial_mask != mask[node] {
            println!(
                "DEBUG: Node {} mask changed: {:02b} -> {:02b}",
                node, initial_mask, mask[node]
            );
        }
    }

    // Calculate shared and union branch lengths (like unifrac_pair)
    let (mut shared, mut union) = (0.0, 0.0);
    println!("DEBUG: Branch analysis for distance calculation:");
    for &node in post {
        let m = mask[node];
        if m == 0 {
            continue; // No presence in either sample
        }
        let len = lens[node] as f64;
        let branch_type = if m == A_BIT {
            "A only"
        } else if m == B_BIT {
            "B only"
        } else {
            "Both A&B"
        };

        println!(
            "DEBUG: Node {} (len={:.3}): mask={:02b} = {}",
            node, len, m, branch_type
        );

        if m == A_BIT || m == B_BIT {
            // Branch present in only one sample
            union += len;
        } else {
            // Branch present in both samples (m == A_BIT | B_BIT)
            shared += len;
            union += len;
        }
    }

    println!(
        "DEBUG: UniFrac calculation - shared: {}, union: {}, distance: {}",
        shared,
        union,
        if union == 0.0 {
            0.0
        } else {
            1.0 - shared / union
        }
    );

    // Return UniFrac distance: 1.0 - shared/union (like unifrac_pair)
    if union == 0.0 {
        0.0
    } else {
        1.0 - shared / union
    }
}

//--------------------------------------------------------------------------------------//
// Sparsity-aware helper functions
//--------------------------------------------------------------------------------------//

/// Identify leaf node indices that are present in either sample A or B (unweighted)
#[allow(dead_code)] // Will be used in Ticket 3 (sparse traversal)
fn identify_relevant_leaves(
    leaf_ids: &[usize],
    a: &BitVec<u8, bitvec::order::Lsb0>,
    b: &BitVec<u8, bitvec::order::Lsb0>,
) -> HashSet<usize> {
    let mut relevant = HashSet::new();
    for (leaf_pos, &leaf_id) in leaf_ids.iter().enumerate() {
        if leaf_pos < a.len() && leaf_pos < b.len() {
            if a[leaf_pos] || b[leaf_pos] {
                relevant.insert(leaf_id);
            }
        }
    }
    relevant
}

/// Identify leaf node indices that are present in either sample A or B (weighted)
#[allow(dead_code)] // Will be used in Ticket 3 (sparse traversal)
fn identify_relevant_leaves_weighted(
    leaf_ids: &[usize],
    va: &[f32],
    vb: &[f32],
) -> HashSet<usize> {
    let mut relevant = HashSet::new();
    for (leaf_pos, &leaf_id) in leaf_ids.iter().enumerate() {
        if leaf_pos < va.len() && leaf_pos < vb.len() {
            if va[leaf_pos] > 0.0 || vb[leaf_pos] > 0.0 {
                relevant.insert(leaf_id);
            }
        }
    }
    relevant
}

/// Mark all ancestors of relevant leaves using bottom-up traversal
/// Returns: Vec<bool> where is_relevant[i] = true if node i is relevant
/// 
/// This function traverses upward from each relevant leaf to the root,
/// marking all ancestors. It uses early termination optimization: if a
/// parent is already marked, all ancestors above it are already marked.
#[allow(dead_code)] // Will be used in Ticket 4 (sparse traversal)
fn mark_relevant_ancestors(
    relevant_leaves: &HashSet<usize>,
    parent: &[usize],
    root_idx: usize,
) -> Vec<bool> {
    let num_nodes = parent.len();
    let mut is_relevant = vec![false; num_nodes];
    
    // Mark all relevant leaves
    for &leaf_id in relevant_leaves {
        if leaf_id < num_nodes {
            is_relevant[leaf_id] = true;
        }
    }
    
    // Traverse upward from each leaf to root
    // Optimization: Stop early if parent is already marked (all ancestors above are marked)
    for &leaf_id in relevant_leaves {
        if leaf_id >= num_nodes {
            continue;
        }
        
        let mut current = leaf_id;
        while current != root_idx {
            let parent_id = parent[current];
            
            // Bounds check
            if parent_id >= num_nodes {
                break;
            }
            
            // If parent is already marked, all ancestors above are already marked
            if is_relevant[parent_id] {
                break;
            }
            
            is_relevant[parent_id] = true;
            current = parent_id;
        }
    }
    
    // Always mark root (it's the ancestor of all leaves, even if no relevant leaves exist)
    if root_idx < num_nodes {
        is_relevant[root_idx] = true;
    }
    
    is_relevant
}

//--------------------------------------------------------------------------------------//
// Fast unweighted UniFrac using bit masks
//--------------------------------------------------------------------------------------//

/// Fast unweighted UniFrac using bit masks
fn unifrac_pair(
    post: &[usize],
    kids: &[Vec<usize>],
    lens: &[f32],
    leaf_ids: &[usize],
    a: &BitVec<u8, bitvec::order::Lsb0>,
    b: &BitVec<u8, bitvec::order::Lsb0>,
) -> f64 {
    const A_BIT: u8 = 0b01;
    const B_BIT: u8 = 0b10;
    let mut mask = vec![0u8; lens.len()];
    for (leaf_pos, &nid) in leaf_ids.iter().enumerate() {
        if a[leaf_pos] {
            mask[nid] |= A_BIT;
        }
        if b[leaf_pos] {
            mask[nid] |= B_BIT;
        }
    }
    for &v in post {
        for &c in &kids[v] {
            mask[v] |= mask[c];
        }
    }
    let (mut shared, mut union) = (0.0, 0.0);
    for &v in post {
        let m = mask[v];
        if m == 0 {
            continue;
        }
        let len = lens[v] as f64;
        if m == A_BIT || m == B_BIT {
            union += len;
        } else {
            shared += len;
            union += len;
        }
    }
    if union == 0.0 {
        0.0
    } else {
        1.0 - shared / union
    }
}

pub(crate) fn unifrac_pair_weighted(
    post: &[usize],
    kids: &[Vec<usize>],
    lens: &[f32],
    leaf_ids: &[usize],
    va: &[f32],
    vb: &[f32],
) -> f64 {
    let sum_a: f32 = va.iter().sum();
    let normalized_a: Vec<f32> = if sum_a > 0.0 {
        let inv_a = 1.0 / sum_a;
        va.iter().map(|&x| x * inv_a).collect()
    } else {
        vec![0.0; va.len()]
    };
    let sum_b: f32 = vb.iter().sum();
    let normalized_b: Vec<f32> = if sum_b > 0.0 {
        let inv_b = 1.0 / sum_b;
        vb.iter().map(|&x| x * inv_b).collect()
    } else {
        vec![0.0; vb.len()]
    };

    let num_nodes = lens.len();
    let mut partial_sums = vec![0.0; num_nodes];

    for (i, &leaf_id) in leaf_ids.iter().enumerate() {
        if i < normalized_a.len() && i < normalized_b.len() {
            let diff = normalized_a[i] - normalized_b[i];
            if diff.abs() > 1e-12 {
                partial_sums[leaf_id] = diff;
            }
        }
    }
    for &v in post {
        for &c in &kids[v] {
            partial_sums[v] += partial_sums[c];
        }
    }

    let mut distance = 0.0f64;
    let mut total_length = 0.0f64;

    for &node_id in post {
        let diff = partial_sums[node_id] as f64;
        let branch_len = lens[node_id] as f64;

        if branch_len > 0.0 {
            distance += diff.abs() * branch_len;
            total_length += branch_len;
        }
    }

    if total_length > 0.0 {
        distance / total_length
    } else {
        0.0
    }
}
// --- SuccTrav and collect_children code copied from unifrac_bp ---

/// SuccTrav for BalancedParensTree building - stores Newick node ID as label
#[allow(dead_code)]
struct SuccTrav<'a> {
    t: &'a NewickTree,
    stack: Vec<(usize, usize, usize)>,
    lens: &'a mut Vec<f32>,
}

impl<'a> SuccTrav<'a> {
    #[allow(dead_code)]
    fn new(t: &'a NewickTree, lens: &'a mut Vec<f32>) -> Self {
        Self {
            t,
            stack: vec![(t.root(), 0, 0)],
            lens,
        }
    }
}

impl<'a> DepthFirstTraverse for SuccTrav<'a> {
    type Label = usize; // Store Newick node ID as label

    fn next(&mut self) -> Option<VisitNode<Self::Label>> {
        let (id, lvl, nth) = self.stack.pop()?;
        let n_children = self.t[id].children().len();
        for (k, &c) in self.t[id].children().iter().enumerate().rev() {
            let nth = n_children - 1 - k;
            self.stack.push((c, lvl + 1, nth));
        }
        if self.lens.len() <= id {
            self.lens.resize(id + 1, 0.0);
        }
        self.lens[id] = self.t[id].branch().copied().unwrap_or(0.0);
        Some(VisitNode::new(id, lvl, nth)) // Return Newick node ID as label
    }
}

#[allow(dead_code)]
fn collect_children<N: succparen::bitwise::ops::NndOne>(
    node: &BpNode<LabelVec<usize>, N, &BalancedParensTree<LabelVec<usize>, N>>,
    kids: &mut [Vec<usize>],
    post: &mut Vec<usize>,
) {
    let pid = node.id() as usize;
    for edge in node.children() {
        let cid = edge.node.id() as usize;
        kids[pid].push(cid);
        collect_children(&edge.node, kids, post);
    }
    post.push(pid);
}

// End of NewDistUniFrac

/// Build a bitmask for data[i] > 0.0 using AVX2 intrinsics.
/// # Safety
/// - This function is marked `unsafe` because it uses `target_feature(enable = "avx2")`
///   and raw pointer arithmetic.
/// - The caller must ensure the CPU supports AVX2.
///
/// # Parameters
/// - `data`: slice of f32 values (length can be very large).
/// # Returns
/// - A `Vec<u64>` with `(data.len() + 63) / 64` elements.
///   Each bit in the `u64` corresponds to one element in `data`,
///   set if `data[i] > 0.0`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn make_presence_mask_f32_avx2(data: &[f32]) -> Vec<u64> {
    let len = data.len();
    let num_chunks = (len + 63) / 64;
    let mut mask = vec![0u64; num_chunks];
    // We'll process 8 floats at a time (AVX2 = 256 bits).
    const STRIDE: usize = 8;
    let blocks = len / STRIDE;
    let remainder = len % STRIDE;
    let ptr = data.as_ptr();
    // For each block of 8 floats
    for blk_idx in 0..blocks {
        let offset = blk_idx * STRIDE;
        // Load 8 consecutive f32 values
        // _mm256_loadu_ps => unaligned load is usually fine on modern x86
        let v = _mm256_loadu_ps(ptr.add(offset));
        // Compare each float in `v` to 0.0 => result bits set to 1 if > 0.0
        let gt_mask = _mm256_cmp_ps(v, _mm256_set1_ps(0.0), _CMP_GT_OQ);
        // `_mm256_movemask_ps` extracts the top bit of each float comparison
        // into an 8-bit integer: 1 bit per float, 0..7
        let bitmask = _mm256_movemask_ps(gt_mask) as u32; // 8 bits used

        // Now we need to place each bit into the appropriate position in `mask`.
        // The i-th bit in `bitmask` corresponds to data[offset + i].
        // So for i in 0..8, if that bit is set, set the corresponding bit in `mask`.
        if bitmask == 0 {
            // No bits set in this block => skip
            continue;
        }
        for i in 0..STRIDE {
            let global_idx = offset + i;
            // Check if bit i is set
            if (bitmask & (1 << i)) != 0 {
                let chunk_idx = global_idx / 64;
                let bit_idx = global_idx % 64;
                mask[chunk_idx] |= 1 << bit_idx;
            }
        }
    }
    // Handle any leftover floats at the tail
    let tail_start = blocks * STRIDE;
    if remainder > 0 {
        for i in tail_start..(tail_start + remainder) {
            if *data.get_unchecked(i) > 0.0 {
                let chunk_idx = i / 64;
                let bit_idx = i % 64;
                mask[chunk_idx] |= 1 << bit_idx;
            }
        }
    }
    mask
}

/// Fallback scalar version, for reference.
fn make_presence_mask_f32_scalar(data: &[f32]) -> Vec<u64> {
    let num_chunks = (data.len() + 63) / 64;
    let mut mask = vec![0u64; num_chunks];
    for (i, &val) in data.iter().enumerate() {
        if val > 0.0 {
            let chunk_idx = i / 64;
            let bit_idx = i % 64;
            mask[chunk_idx] |= 1 << bit_idx;
        }
    }
    mask
}

fn make_presence_mask_f32(data: &[f32]) -> Vec<u64> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { make_presence_mask_f32_avx2(data) }
        } else {
            make_presence_mask_f32_scalar(data)
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        make_presence_mask_f32_scalar(data)
    }
}

fn or_masks(a: &[u64], b: &[u64]) -> Vec<u64> {
    a.iter().zip(b.iter()).map(|(x, y)| x | y).collect()
}

fn extract_set_bits(m: &[u64]) -> Vec<usize> {
    let mut indices = Vec::new();
    for (chunk_idx, &chunk) in m.iter().enumerate() {
        if chunk == 0 {
            continue;
        }
        let base = chunk_idx * 64;
        let mut c = chunk;
        while c != 0 {
            let bit_index = c.trailing_zeros() as usize;
            indices.push(base + bit_index);
            c &= !(1 << bit_index);
        }
    }
    indices
}

//--------------------------------------------------------------------------------------//
//  Bitwise skip-zero logic for unweighted
//--------------------------------------------------------------------------------------//

fn compute_unifrac_for_pair_unweighted_bitwise(
    tint: &[usize],
    lint: &[f32],
    nodes_in_order: &[usize],
    leaf_map: &HashMap<String, usize>,
    feature_names: &[String],
    va: &[f32],
    vb: &[f32],
) -> f32 {
    debug!("=== compute_unifrac_for_pair_unweighted_bitwise ===");
    let num_nodes = nodes_in_order.len();
    let mut partial_sums = vec![0.0; num_nodes];

    // 1) Build presence masks for va, vb
    let mask_a = make_presence_mask_f32(va);
    let mask_b = make_presence_mask_f32(vb);

    // 2) Combine => find non-zero indices
    let combined = or_masks(&mask_a, &mask_b);
    let non_zero_indices = extract_set_bits(&combined);
    debug!("non_zero_indices = {:?}", non_zero_indices);

    // 3) Create local arrays
    let mut local_a = Vec::with_capacity(non_zero_indices.len());
    let mut local_b = Vec::with_capacity(non_zero_indices.len());
    let mut local_feats = Vec::with_capacity(non_zero_indices.len());

    for &idx in &non_zero_indices {
        local_a.push(va[idx]);
        local_b.push(vb[idx]);
        local_feats.push(&feature_names[idx]);
    }

    // 4) Convert to presence/absence, then sum
    let mut sum_a = 0.0;
    let mut sum_b = 0.0;
    for i in 0..local_a.len() {
        if local_a[i] > 0.0 {
            local_a[i] = 1.0;
            sum_a += 1.0;
        } else {
            local_a[i] = 0.0;
        }
        if local_b[i] > 0.0 {
            local_b[i] = 1.0;
            sum_b += 1.0;
        } else {
            local_b[i] = 0.0;
        }
    }
    // Normalize each sample
    if sum_a > 0.0 {
        let inv_a = 1.0 / sum_a;
        for x in local_a.iter_mut() {
            *x *= inv_a;
        }
    }
    if sum_b > 0.0 {
        let inv_b = 1.0 / sum_b;
        for x in local_b.iter_mut() {
            *x *= inv_b;
        }
    }

    // 5) partial sums => difference
    for (i, feat_name) in local_feats.iter().enumerate() {
        if let Some(&leaf_idx) = leaf_map.get(*feat_name) {
            let diff = local_a[i] - local_b[i];
            if diff.abs() > 1e-12 {
                partial_sums[leaf_idx] = diff;
                debug!(
                    "  => partial_sums[leaf_idx={}] = {} for feat='{}'",
                    leaf_idx, diff, feat_name
                );
            }
        }
    }

    // 6) Propagate partial sums up the chain
    let mut dist = 0.0;
    for i in 0..(num_nodes - 1) {
        let val = partial_sums[i];
        partial_sums[tint[i]] += val;
        dist += lint[i] * val.abs();
    }
    debug!("Final unweighted dist={}", dist);
    dist
}

//--------------------------------------------------------------------------------------//
//  Bitwise skip-zero logic for weighted
//--------------------------------------------------------------------------------------//

fn compute_unifrac_for_pair_weighted_bitwise(
    tint: &[usize],
    lint: &[f32],
    nodes_in_order: &[usize],
    leaf_map: &HashMap<String, usize>,
    feature_names: &[String],
    va: &[f32],
    vb: &[f32],
) -> f32 {
    debug!("=== compute_unifrac_for_pair_weighted_bitwise ===");
    let num_nodes = nodes_in_order.len();
    let mut partial_sums = vec![0.0; num_nodes];

    // 1) presence masks
    let mask_a = make_presence_mask_f32(va);
    let mask_b = make_presence_mask_f32(vb);

    // 2) combine
    let combined = or_masks(&mask_a, &mask_b);
    let non_zero_indices = extract_set_bits(&combined);
    debug!("non_zero_indices = {:?}", non_zero_indices);

    // 3) build local arrays
    let mut local_a = Vec::with_capacity(non_zero_indices.len());
    let mut local_b = Vec::with_capacity(non_zero_indices.len());
    let mut local_feats = Vec::with_capacity(non_zero_indices.len());
    for &idx in &non_zero_indices {
        local_a.push(va[idx]);
        local_b.push(vb[idx]);
        local_feats.push(&feature_names[idx]);
    }

    // 4) sum & normalize
    let sum_a: f32 = local_a.iter().sum();
    if sum_a > 0.0 {
        let inv_a = 1.0 / sum_a;
        for x in local_a.iter_mut() {
            *x *= inv_a;
        }
    }
    let sum_b: f32 = local_b.iter().sum();
    if sum_b > 0.0 {
        let inv_b = 1.0 / sum_b;
        for x in local_b.iter_mut() {
            *x *= inv_b;
        }
    }

    // 5) partial sums => diff
    for (i, feat_name) in local_feats.iter().enumerate() {
        if let Some(&leaf_idx) = leaf_map.get(*feat_name) {
            let diff = local_a[i] - local_b[i];
            if diff.abs() > 1e-12 {
                partial_sums[leaf_idx] = diff;
                debug!(
                    "   => partial_sums[leaf_idx={}] = {} for feat='{}'",
                    leaf_idx, diff, feat_name
                );
            }
        }
    }

    // 6) propagate partial sums
    let mut dist = 0.0;
    for i in 0..(num_nodes - 1) {
        let val = partial_sums[i];
        partial_sums[tint[i]] += val;
        dist += lint[i] * val.abs();
    }
    debug!("Final weighted dist={}", dist);
    dist
}

//DistUniFrac_C
// Demonstration of calling `one_dense_pair_v2t` from Rust with tests
// -----------------------------------------------------------------------------
// 1) Reproduce some enums / types from C++ side
// -----------------------------------------------------------------------------
/// Mirror of your `compute_status` enum from C++.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ComputeStatus {
    Okay = 0,
    UnknownMethod = 1,
    TreeMissing = 2,
    TableMissing = 3,
    TableEmpty = 4,
    TableAndTreeDoNotOverlap = 5,
    OutputError = 6,
    InvalidMethod = 7,
    GroupingMissing = 8,
}

/// Opaque BPTree pointer type; the actual structure is in C++.
#[repr(C)]
pub struct OpaqueBPTree {
    _private: [u8; 0], // no fields in Rust
}

/// This struct holds parameters for one_dense_pair_v2t in C++.
#[repr(C)]
pub struct DistUniFrac_C {
    pub n_obs: c_uint,
    pub obs_ids: *const *const c_char,
    pub tree_data: *const OpaqueBPTree,
    pub unifrac_method: *const c_char,
    pub variance_adjust: bool,
    pub alpha: c_double,
    pub bypass_tips: bool,
}

// ----------------------------------------------------------------------------
// 2) Expose the extern "C" functions from your C++ library
// ----------------------------------------------------------------------------
extern "C" {
    /// Builds a BPTree from a Newick string.  
    /// On success, it writes an allocated `OpaqueBPTree*` into `tree_data_out`.
    pub fn load_bptree_opaque(newick: *const c_char, tree_data_out: *mut *mut OpaqueBPTree);

    /// Frees the BPTree allocated by `load_bptree_opaque`.
    pub fn destroy_bptree_opaque(tree_data: *mut *mut OpaqueBPTree);

    /// The main distance function in C++:  
    ///   one_dense_pair_v2t(n_obs, obs_ids, sample1, sample2, tree_data, method_str, ...)
    pub fn one_dense_pair_v2t(
        n_obs: c_uint,
        obs_ids: *const *const c_char,
        sample1: *const c_double,
        sample2: *const c_double,
        tree_data: *const OpaqueBPTree,
        unifrac_method: *const c_char,
        variance_adjust: bool,
        alpha: c_double,
        bypass_tips: bool,
        result: *mut c_double,
    ) -> ComputeStatus;
}

// ----------------------------------------------------------------------------
// 3) Provide a function bridging from f32 slices to one_dense_pair_v2t
// ----------------------------------------------------------------------------
// TEMPORARILY COMMENTED OUT - external C UniFrac library functions
/*
#[no_mangle]
pub extern "C" fn dist_unifrac_c(
    ctx: *const DistUniFrac_C,
    va: *const f32,
    vb: *const f32,
    length: c_ulonglong,
) -> f32 {
    if ctx.is_null() {
        eprintln!("dist_unifrac_c: NULL context pointer!");
        return 0.0;
    }
    let ctx_ref = unsafe { &*ctx };

    if length != ctx_ref.n_obs as u64 {
        eprintln!(
            "dist_unifrac_c: length mismatch. Got {}, expected {}",
            length, ctx_ref.n_obs
        );
        return 0.0;
    }

    let slice_a = unsafe { slice::from_raw_parts(va, length as usize) };
    let slice_b = unsafe { slice::from_raw_parts(vb, length as usize) };

    let mut buf_a = Vec::with_capacity(slice_a.len());
    let mut buf_b = Vec::with_capacity(slice_b.len());
    for (&a_val, &b_val) in slice_a.iter().zip(slice_b.iter()) {
        buf_a.push(a_val as f64);
        buf_b.push(b_val as f64);
    }

    let mut dist_out: c_double = 0.0;

unsafe impl Send for DistUniFracCFFI {}
unsafe impl Sync for DistUniFracCFFI {}
*/
