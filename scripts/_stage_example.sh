# Shared staging helper, sourced by the quick-start wrappers (not executable).
#
# stage_example <run>: when <run> is the bundled example, refresh
# logs/g1_flat/example (git-ignored) from the top-level example/ folder.
# The staged copy always MIRRORS the bundled example: example-owned files
# (teacher checkpoint, params/, the example student policy) are overwritten on
# every quick-start run, so updating example/ takes effect immediately. Extra
# files you created in the staged folder (e.g. your own distilled student
# checkpoints) are left in place.
#
# Note: if you distill a student FROM the example run, your exported
# student_checkpoints/student_policy_jit.pt shares its path with the bundled
# example policy and will be overwritten on the next quick-start run — copy it
# out if you want to keep it.
#
# No-op for any other run name (i.e. your own runs).

stage_example() {
    local run="$1"
    if [ "$run" = "example" ] && [ -d "example" ]; then
        mkdir -p "logs/g1_flat/example"
        cp -rf "example/." "logs/g1_flat/example/"
    fi
}
