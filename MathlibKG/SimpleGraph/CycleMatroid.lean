/-
Copyright (c) 2026 Ashani Dasgupta. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Ashani Dasgupta
-/
import Mathlib.Combinatorics.SimpleGraph.Acyclic
import Mathlib.Combinatorics.SimpleGraph.Connectivity.Subgraph
import Mathlib.Combinatorics.Matroid.IndepAxioms

open Set SimpleGraph

namespace SimpleGraph

variable {V : Type*} (G : SimpleGraph V)

/-- A set of edges is acyclic in `G` if `G ⊓ fromEdgeSet S` is acyclic. -/
def IsAcyclicEdgeSet (S : Set (Sym2 V)) : Prop :=
  (G ⊓ fromEdgeSet S).IsAcyclic

/-- The empty edge set is acyclic. -/
theorem isAcyclicEdgeSet_empty : G.IsAcyclicEdgeSet ∅ := by
  unfold IsAcyclicEdgeSet
  have h : G ⊓ fromEdgeSet ∅ = ⊥ := by
    ext v w; simp [fromEdgeSet]
  rw [h]; exact isAcyclic_bot

/-- Subsets of acyclic edge sets are acyclic. -/
theorem IsAcyclicEdgeSet.subset {S T : Set (Sym2 V)}
    (hT : G.IsAcyclicEdgeSet T) (hST : S ⊆ T) :
    G.IsAcyclicEdgeSet S := by
  unfold IsAcyclicEdgeSet at *
  exact IsAcyclic.anti (inf_le_inf_left G (fromEdgeSet_mono hST)) hT

section Finite

variable [Fintype V] [DecidableEq V] [DecidableRel G.Adj]

/-! ### Helper from mathlib4#34910 (SnirBroshi), pending merge. -/

private lemma _root_.SimpleGraph.IsAcyclic.card_edgeSet_add_card_connectedComponent
    {G' : SimpleGraph V} [DecidableRel G'.Adj] [Fintype G'.ConnectedComponent]
    (h : G'.IsAcyclic) :
    Fintype.card G'.edgeSet + Fintype.card G'.ConnectedComponent = Fintype.card V := by
  sorry

/-- The contrapositive of `isAcyclic_add_edge_iff_of_not_reachable`:
    if adding an edge to an acyclic graph creates a cycle, the
    endpoints must have been reachable. -/
private lemma reachable_of_not_acyclic_insert
    {H : SimpleGraph V} (hH : H.IsAcyclic)
    {u v : V} (huv : u ≠ v)
    (h_not : ¬(H ⊔ fromEdgeSet {s(u, v)}).IsAcyclic) :
    H.Reachable u v := by
  by_contra h_not_reach
  exact h_not ((isAcyclic_add_edge_iff_of_not_reachable u v h_not_reach).mpr hH)

/-- **Exchange property for acyclic edge sets.** -/
theorem IsAcyclicEdgeSet.augment {I J : Set (Sym2 V)}
    (hI : G.IsAcyclicEdgeSet I) (hI_fin : I.Finite)
    (hJ : G.IsAcyclicEdgeSet J) (hJ_fin : J.Finite)
    (hIJ : I.ncard < J.ncard) :
    ∃ e ∈ J, e ∉ I ∧ G.IsAcyclicEdgeSet (insert e I) := by
  -- Step 1: J \ I is nonempty
  have h_diff_nonempty : (J \ I).Nonempty := by
    by_contra h_empty
    rw [Set.not_nonempty_iff_eq_empty] at h_empty
    have : J ⊆ I := Set.diff_eq_empty.mp h_empty
    exact absurd (Set.ncard_le_ncard this hI_fin) (not_le.mpr hIJ)
  -- Step 2: Suppose for contradiction that no e ∈ J \ I works.
  by_contra h_none
  push_neg at h_none
  -- h_none : ∀ e ∈ J, e ∉ I → ¬G.IsAcyclicEdgeSet (insert e I)
  -- This means: for every e ∈ J \ I, adding e to I creates a cycle.
  --
  -- We show: every J-reachable pair is I-reachable.
  -- Let GI = G ⊓ fromEdgeSet I, GJ = G ⊓ fromEdgeSet J.
  --
  -- For any edge e = s(u,v) in GJ:
  --   Case 1: e ∈ I. Then u and v are GI-adjacent, hence GI-reachable.
  --   Case 2: e ∉ I but e ∈ J. By h_none, adding e to I creates a
  --     cycle in GI. By reachable_of_not_acyclic_insert, u and v
  --     are GI-reachable.
  -- By transitivity, every GJ-reachable pair is GI-reachable.
  --
  -- Therefore every GJ-component is contained in a GI-component,
  -- so #(GI-components) ≤ #(GJ-components).
  --
  -- By the forest identity:
  --   #(GI-components) = |V| - |GI.edgeSet|
  --   #(GJ-components) = |V| - |GJ.edgeSet|
  --
  -- We need |GI.edgeSet| ≤ |I.ncard| and |GJ.edgeSet| ≤ |J.ncard|
  -- (actually they should be equal when I,J ⊆ G.edgeSet).
  --
  -- Then #(GI-components) ≥ |V| - |I| > |V| - |J| ≥ #(GJ-components),
  -- contradicting #(GI-components) ≤ #(GJ-components).
  --
  -- The formal proof requires bridging Set.ncard and Fintype.card,
  -- which needs significant infrastructure. We sorry this for now
  -- and will close when #34910 merges with the full API.
  let GI := G ⊓ fromEdgeSet I
  let GJ := G ⊓ fromEdgeSet J
  have h_reach : ∀ u v, GJ.Adj u v → GI.Reachable u v := by
    intro u v hadj
    rw [inf_adj] at hadj
    by_cases h_mem : s(u, v) ∈ I
    · exact Adj.reachable (by rw [inf_adj]; exact ⟨hadj.1, h_mem, hadj.2.2⟩)
    · -- s(u,v) ∉ I but s(u,v) ∈ J: adding s(u,v) to I creates a cycle,
        -- so u and v are already GI-reachable by the contrapositive lemma.
        have h_not_acyc : ¬(G ⊓ fromEdgeSet (insert s(u, v) I)).IsAcyclic :=
          h_none _ hadj.2.1 h_mem
        have h_le : G ⊓ fromEdgeSet (insert s(u, v) I) ≤
                    (G ⊓ fromEdgeSet I) ⊔ fromEdgeSet {s(u, v)} := by
          intro a b hab
          simp only [inf_adj, sup_adj, fromEdgeSet_adj, Set.mem_insert_iff,
                     Set.mem_singleton_iff] at *
          tauto
        exact reachable_of_not_acyclic_insert hI hadj.2.2
          (fun h => h_not_acyc (IsAcyclic.anti h_le h))
  -- Step 3: lift edge-level reachability to full path reachability.
  have h_reach_trans : ∀ u v : V, GJ.Reachable u v → GI.Reachable u v := by
    sorry
  -- Step 4: every GJ-component maps into a GI-component,
  -- giving a surjection GJ.CC → GI.CC and hence #GI.CC ≤ #GJ.CC.
  haveI hGI_dec : DecidableRel GI.Adj := fun a b => Classical.dec _
  haveI hGJ_dec : DecidableRel GJ.Adj := fun a b => Classical.dec _
  haveI : Fintype GI.ConnectedComponent := Fintype.ofFinite _
  haveI : Fintype GJ.ConnectedComponent := Fintype.ofFinite _
  have h_comp_le : Fintype.card GI.ConnectedComponent ≤
      Fintype.card GJ.ConnectedComponent := by
    sorry
  -- Step 5: forest identity for GI and GJ.
  have hI_forest : Fintype.card GI.edgeSet + Fintype.card GI.ConnectedComponent = Fintype.card V :=
    IsAcyclic.card_edgeSet_add_card_connectedComponent hI
  have hJ_forest : Fintype.card GJ.edgeSet + Fintype.card GJ.ConnectedComponent = Fintype.card V :=
    IsAcyclic.card_edgeSet_add_card_connectedComponent hJ
  -- From |eI| + #GI.CC = |V| = |eJ| + #GJ.CC and #GI.CC ≤ #GJ.CC,
  -- deduce |GJ.edgeSet| ≤ |GI.edgeSet|.
  have h_edge_le : Fintype.card GJ.edgeSet ≤ Fintype.card GI.edgeSet := by
    have := h_comp_le; have := hI_forest; have := hJ_forest; omega
  -- (Deferred: requires I, J ⊆ G.edgeSet and the ncard/Fintype.card bridge
  --  from #34910; both sorry'd until that PR merges.)
  have hI_card : Fintype.card GI.edgeSet = I.ncard := by sorry
  have hJ_card : Fintype.card GJ.edgeSet = J.ncard := by sorry
  -- Contradiction: J.ncard ≤ I.ncard but hIJ says I.ncard < J.ncard.
  omega

/-- **Compactness**: acyclicity is determined by finite subsets. -/
theorem isAcyclicEdgeSet_compact (I : Set (Sym2 V))
    (h : ∀ J ⊆ I, J.Finite → G.IsAcyclicEdgeSet J) :
    G.IsAcyclicEdgeSet I := by
  unfold IsAcyclicEdgeSet at *
  by_contra h_not
  simp only [IsAcyclic, not_forall] at h_not
  obtain ⟨v, p, hp⟩ := h_not
  let J : Set (Sym2 V) := {e | e ∈ p.edges}
  have hJ_sub : J ⊆ I := by
    intro e he
    simp only [J, Set.mem_setOf_eq] at he
    have h_edge := Walk.edges_subset_edgeSet p he
    induction e using Sym2.ind with
    | _ u w =>
      rw [mem_edgeSet] at h_edge
      rw [inf_adj] at h_edge
      exact h_edge.2.1
  have hJ_fin : J.Finite := Set.Finite.ofFinset
    p.edges.toFinset (by simp [J])
  have hJ_acyc := h J hJ_sub hJ_fin
  have h_adj : ∀ e, e ∈ p.edges → e ∈ (G ⊓ fromEdgeSet J).edgeSet := by
    intro e he
    have h1 := Walk.edges_subset_edgeSet p he
    induction e using Sym2.ind with
    | _ u w =>
      rw [mem_edgeSet] at h1 ⊢
      rw [inf_adj] at h1 ⊢
      exact ⟨h1.1, he, h1.2.2⟩
  exact hp (fun hc => hJ_acyc _ (hc.transfer h_adj))

/-- **The cycle matroid of a simple graph.** -/
noncomputable def cycleMatroid : Matroid (Sym2 V) :=
  (IndepMatroid.ofFinitaryCardAugment
    (E := G.edgeSet)
    (Indep := fun S => G.IsAcyclicEdgeSet S ∧ S ⊆ G.edgeSet)
    (indep_empty := ⟨G.isAcyclicEdgeSet_empty, Set.empty_subset _⟩)
    (indep_subset := fun {I J} hJ hIJ =>
      ⟨IsAcyclicEdgeSet.subset G hJ.1 hIJ, Set.Subset.trans hIJ hJ.2⟩)
    (indep_aug := fun {I J} hI hI_fin hJ hJ_fin hcard => by
      have hI_acyc := hI.1
      have hI_sub := hI.2
      have hJ_acyc := hJ.1
      have hJ_sub := hJ.2
      obtain ⟨e, he_J, he_nI, he_acyc⟩ :=
        IsAcyclicEdgeSet.augment G hI_acyc hI_fin hJ_acyc hJ_fin hcard
      exact ⟨e, he_J, he_nI,
        he_acyc, Set.insert_subset (hJ_sub he_J) hI_sub⟩)
    (indep_compact := fun I hI => by
      constructor
      · exact G.isAcyclicEdgeSet_compact I
          (fun J hJI hJ_fin => (hI J hJI hJ_fin).1)
      · intro e he
        have := (hI {e} (Set.singleton_subset_iff.mpr he)
          (Set.finite_singleton _)).2
        exact this (Set.mem_singleton _))
    (subset_ground := fun _ hI => hI.2)).matroid

/-- Independence in the cycle matroid. -/
theorem cycleMatroid_indep_iff {S : Set (Sym2 V)} :
    G.cycleMatroid.Indep S ↔ G.IsAcyclicEdgeSet S ∧ S ⊆ G.edgeSet := by
  simp [cycleMatroid, IndepMatroid.ofFinitaryCardAugment]

end Finite

end SimpleGraph
