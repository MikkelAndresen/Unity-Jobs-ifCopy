# FoW - Unity Project

Unity 6 (6000.4.4f1) project using URP.

## Enter Play Mode Settings

This project has **Domain Reload** and **Scene Reload** disabled on enter play mode (Project Settings > Editor > Enter Play Mode Settings).

### Consequences for code

**Static fields are NOT reset** between play mode sessions. Any static variable that holds runtime state will carry over from the previous play session. You must reset statics manually:

```csharp
[RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
static void ResetStatics()
{
    // Reset all static fields here
    s_instance = null;
    s_initialized = false;
}
```

**Singletons require explicit cleanup.** MonoBehaviour singletons using a static `Instance` field will not be nulled out. Use `[RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]` to clear the static reference.

**Static events accumulate subscribers.** If code subscribes to a static event in `OnEnable`/`Awake` without unsubscribing, handlers will stack up across play sessions. Always unsubscribe in `OnDisable`/`OnDestroy`, or clear the event in a `[RuntimeInitializeOnLoadMethod]`.

**Scene state persists.** Because scene reload is disabled, `Awake`/`Start`/`OnEnable` will NOT re-fire for objects already in the scene. Only newly instantiated objects get their lifecycle callbacks. Do not rely on `Awake`/`Start` for per-session initialization on scene objects.

**`SceneManager.sceneLoaded` does not fire** for the initial scene on play mode entry (since it was never unloaded). Code that depends on this callback for setup needs an alternative path.

### What to watch for when writing new code

- Never assume static state is clean at play mode start
- Always pair event subscriptions with unsubscriptions
- Prefer `[RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]` for resetting statics over `Awake`/`Start`
- Use `[RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]` or `AfterSceneLoad` for initialization that must happen every play session
- Test code by entering play mode multiple times in a row without stopping — bugs from stale state often only appear on the second run

## Project Structure

- `Assets/ClaudeTools/Editor/` - Build and test runner utilities (used via Unity MCP)
- `Assets/Tests/Editor/` - EditMode tests (NUnit)
- `Assets/Tests/PlayMode/` - PlayMode tests (UnityTest)

## Running Tests

Tests can be run via the Unity MCP `Unity_RunCommand` tool using the helpers in `Assets/ClaudeTools/Editor/RunTests.cs`:
- `ClaudeTools.RunTests.RunAllEditModeTests()` — runs all EditMode tests
- `ClaudeTools.RunTests.RunAllPlayModeTests()` — runs all PlayMode tests
- `ClaudeTools.RunTests.RunSmokeTests()` — runs Smoke category tests only

These helpers print a compact summary to the console, reducing token usage compared to using `TestRunnerApi` directly. Read results afterwards with `Unity_ReadConsole`. If `RunTests.cs` doesn't cover your use case (e.g. running a specific test class or filtering by name), fall back to `TestRunnerApi` directly — but consider whether the missing functionality should be added to `RunTests.cs` for future use.

- PlayMode tests use `[UnityPlatform(RuntimePlatform.WindowsEditor, RuntimePlatform.OSXEditor, RuntimePlatform.LinuxEditor)]` — they are editor-only and will be skipped in standalone player builds

## Building

Build profiles are in `Assets/Settings/Build Profiles/`. Available profiles: `macOS`, `macOS_dev`.

To build via `Unity_RunCommand`, use the `ClaudeTools.BuildWithProfile.Build()` helper:

```csharp
using UnityEditor;
internal class CommandScript : IRunCommand {
    public void Execute(ExecutionResult result) {
        ClaudeTools.BuildWithProfile.Build("macOS");
    }
}
```

`Build(profileName)` resolves `Assets/Settings/Build Profiles/{profileName}.asset` and outputs to `Builds/{profileName}/FoW.app`. Optional overrides: `Build(profileName, outputPath, extraOptions)`.

## Verification Before Testing

When you believe code changes are ready for testing (not after every individual edit), verify the project by following these steps in order:

1. **Check compilation**: First, trigger an asset refresh via `Unity_RunCommand` by calling `AssetDatabase.Refresh()` — Unity does not always detect external file changes automatically. Then use `Unity_ReadConsole` with `Types: ["Error"]` to confirm there are no compilation errors.
2. **Run Editor tests**: If compilation succeeds, run all EditMode tests via `Unity_RunCommand` by calling `ClaudeTools.RunTests.RunAllEditModeTests()`. This helper prints a compact summary to the console, reducing token usage compared to using `TestRunnerApi` directly. After running, read results from the console with `Unity_ReadConsole`. Do not consider the work complete until both compilation and tests pass.
