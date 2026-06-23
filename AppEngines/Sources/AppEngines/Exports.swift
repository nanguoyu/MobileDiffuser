// On macOS, re-export the FLUX engine so the app gets `Flux2FacadeEngine` via `import AppEngines`.
// On iOS this module is intentionally empty — FLUX is excluded until it is ported (cross-platform
// image handling + partial load). See docs/BLUEPRINT.md ("FLUX on iOS").
#if os(macOS)
@_exported import Flux2DiffusionEngine
#endif
