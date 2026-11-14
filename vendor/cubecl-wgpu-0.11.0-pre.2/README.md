# CubeCL WGPU Runtime

[CubeCL](https://github.com/tracel-ai/cubecl) WGPU runtime.

## Configuration

You can set `CUBECL_WGPU_MAX_TASKS` to a positive integer that determines how many computing tasks are submitted in batches to the graphics API.

Two further caps (issue 1217 of the vendoring consumer) pace the host against the GPU; both keep their upstream hard-coded values when unset:

- `CUBECL_WGPU_MAX_TOTAL_TASKS` (default 512) — in-flight submission-task cap; reaching it poll(Wait)s on the previous submission index before more work is encoded.
- `CUBECL_WGPU_MAX_PENDING_WRITES` (default 64) — pending `write_buffer` count that triggers a write-flush submit + poll(Wait) (staging-pool exhaustion guard).

## Platform Support

| Option    | CPU | GPU | Linux | MacOS | Windows | Android | iOS | WASM |
| :-------- | :-: | :-: | :---: | :---: | :-----: | :-----: | :-: | :--: |
| Metal     | No  | Yes |  No   |  Yes  |   No    |   No    | Yes |  No  |
| Vulkan    | Yes | Yes |  Yes  |  Yes  |   Yes   |   Yes   | Yes |  No  |
| OpenGL    | No  | Yes |  Yes  |  Yes  |   Yes   |   Yes   | Yes |  No  |
| WebGpu    | No  | Yes |  No   |  No   |   No    |   No    | No  | Yes  |
| Dx11/Dx12 | No  | Yes |  No   |  No   |   Yes   |   No    | No  |  No  |
