import Mathlib
import Lean

open Lean

/--
  Sanitizes a string for JSON by escaping backslashes and quotes.
  This fixes the "Invalid \escape" error in Python.
-/
def sanitize (s : String) : String :=
  -- We must escape backslash FIRST, then quotes.
  let s1 := s.replace "\\" "\\\\"
  let s2 := s1.replace "\"" "\\\""
  let s3 := s2.replace "\n" " " -- Remove newlines just in case
  s3

/--
  Helper function to extract all dependencies.
-/
def getDeps (info : ConstantInfo) : Array String :=
  let empty : NameSet := {}

  -- 1. Collect from Type
  let set := info.type.foldConsts empty (fun n acc => acc.insert n)

  -- 2. Collect from Value
  let set := match info.value? with
    | some v => v.foldConsts set (fun n acc => acc.insert n)
    | none => set

  -- 3. Convert to Array of Strings
  set.foldl (fun acc n => acc.push (toString n)) #[]

def main : IO Unit := do
  initSearchPath (← findSysroot)
  IO.println "Loading Mathlib environment... (this takes 10-20s)"
  let mods := #[`Mathlib]
  let env ← importModules (mods.map fun m => { module := m }) {}

  let outputPath := "mathlib_graph.json"
  let handle ← IO.FS.Handle.mk outputPath IO.FS.Mode.write
  handle.putStrLn "["

  let mut first := true
  let mut count := 0

  let constants := env.constants.toList

  for (name, info) in constants do
    if name.isInternal || isPrivateName name then continue

    let typeStr := match info with
      | ConstantInfo.thmInfo _ => "theorem"
      | ConstantInfo.defnInfo _ => "definition"
      | ConstantInfo.axiomInfo _ => "axiom"
      | _ => "other"

    if typeStr == "other" then continue

    let deps := getDeps info

    -- Format dependencies with SANITIZATION
    let depsJson := deps.foldl (fun acc dep =>
      let cleanDep := sanitize dep -- <--- THE FIX
      if acc == "[" then acc ++ "\"" ++ cleanDep ++ "\""
      else acc ++ ", \"" ++ cleanDep ++ "\"") "["
    let finalDepsJson := depsJson ++ "]"

    -- Sanitize the node name too
    let cleanName := sanitize (toString name)
    let entry := s!"  \{\"name\": \"{cleanName}\", \"type\": \"{typeStr}\", \"edges\": {finalDepsJson}}"

    if first then
      handle.putStrLn entry
      first := false
    else
      handle.putStrLn (",\n" ++ entry)

    count := count + 1
    if count % 10000 == 0 then
      IO.println s!"Processed {count} declarations..."

  handle.putStrLn "]"
  IO.println s!"\nDone! Graph saved to {outputPath}. Total nodes: {count}"
