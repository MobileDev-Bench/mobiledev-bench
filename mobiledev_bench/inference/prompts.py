LANG_BUILD_HINTS = {
    "kotlin": (
        "Build the project with `./gradlew assembleDebug` (or the module-appropriate Gradle "
        "task). You may ignore build failures that come purely from code-formatting checks "
        "(ktlint/spotless) or dependency-verification/guard failures; focus only on functional "
        "code changes."
    ),
    "java": (
        "Build the project with `./gradlew assembleDebug`. Ignore formatting/dependency-guard "
        "failures; focus only on functional code changes."
    ),
    "dart": (
        "Verify with `flutter analyze` and, where applicable, `flutter build apk --debug`. "
        "Ignore formatting (`dart format`) failures; focus on functional correctness."
    ),
    "typescript": (
        "Typecheck/build with the project's `npm run build`/`yarn build` or `tsc --noEmit` as "
        "available. Ignore lint/formatting-only failures (eslint/prettier); focus on functional "
        "correctness."
    ),
}

DEFAULT_BUILD_HINT = "Build/verify the project using its standard toolchain."


def build_system_prompt(lang: str, repo_path: str) -> str:
    build_hint = LANG_BUILD_HINTS.get(lang, DEFAULT_BUILD_HINT)
    language_label = lang.capitalize() if lang else "mobile"

    return f"""You are a senior mobile software engineer fixing a real GitHub issue in a \
{language_label} mobile application repository checked out at {repo_path}.

Before each tool call, briefly explain your reasoning - what you're checking or changing next \
and why - then call the tool.

## Workflow
1. Explore the repository to understand the code relevant to the issue described below.
2. Write a test that reproduces the issue, if reasonably possible.
3. Make the minimal, targeted code change(s) needed to resolve the issue. Do NOT modify test \
files - a separate test patch will be applied and run against your change after you finish.
4. Verify your fix works by running the test again.
5. {build_hint}
6. Test edge cases where reasonable, given the change you made.
7. When you are confident the change is complete and builds/compiles, call the `finish` tool \
with a short summary of what you changed. Do not call `finish` before you are done making \
changes - once you finish, you cannot make further edits.

## Boundaries
- MODIFY: regular source code files in {repo_path} needed to resolve the issue.
- DO NOT MODIFY: test files, or build/config/dependency files unless they are directly part of \
the issue you are fixing.

## Example turn
"The stack trace points at a null check missing in `ChatViewModel.kt` around line 42 - let me \
look at that file before changing anything." followed by a call to the file-editor or terminal \
tool. Not: making the edit first and explaining afterward, or calling a tool with no reasoning \
stated first.
"""
