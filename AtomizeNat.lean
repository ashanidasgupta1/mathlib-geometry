import Lean
import Lean.Data.Json
import Mathlib

open Lean Meta

/-- Recursive function to grab internal expression logic -/
partial def exprToEdges (e : Expr) : MetaM (List String) := do
  match e with
  | Expr.app f a =>
    let r1 ← exprToEdges f
    let r2 ← exprToEdges a
    return (toString f) :: (toString a) :: r1 ++ r2
  | Expr.lam _ t b _ =>
    let r1 ← exprToEdges t
    let r2 ← exprToEdges b
    return r1 ++ r2
  | Expr.forallE _ t b _ =>
    let r1 ← exprToEdges t
    let r2 ← exprToEdges b
    return r1 ++ r2
  | _ => return []

unsafe def main : IO Unit := do
  let opts : Options := {}
  -- Standard import for Lean 4.x
  let env ← importModules [{module := `Mathlib}] opts
  let coreCtx : Core.Context := { fileName := "<atomizer>", fileMap := { source := "", positions := #[] }, options := opts }

  let jsonEntries ← Core.CoreM.run' (Meta.MetaM.run' (do
    let mut entries : List Json := []
    let constants := (← getEnv).constants.map₁.toList
    for (name, cinfo) in constants do
      if name.toString.startsWith "Nat" then
        let edges ← exprToEdges cinfo.type
        -- Convert List to Array for Json processing
        let edgesArray := edges.toArray.map Json.str
        let entry := Json.mkObj [
          ("name", Json.str name.toString),
          ("edges", Json.arr edgesArray)
        ]
        entries := entry :: entries
    return entries
  )) coreCtx { env := env }

  let outPath := "analysis/nat_atomized_graph.json"
  IO.FS.writeFile outPath (Json.arr jsonEntries.toArray).compress
  IO.println s!"Successfully atomized Nat namespace into {outPath}"
