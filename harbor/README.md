# Harbor tasks

Benchmark tasks packaged as [Harbor](https://github.com/harbor-framework/harbor) tasks, to evaluate agents on brain MRI prediction. The agent gets train images and labels plus test images, and writes predictions. It has no GPU and no internet access.

Each task folder holds its `instruction.md`. `task-template/` holds the shared environment, verifier and config, and `score/` the scorer for each task type.

## Build

```bash
python harbor/build_tasks.py ixi_age
```

This writes `$BRAINMARKS_SMRI_DATA/harbor/ixi_age/`, with the images baked into the environment. The sandbox test split is the benchmark val split.

## Run

On a host with Docker:

```bash
uv tool install harbor
T=$BRAINMARKS_SMRI_DATA/harbor/ixi_age

# Oracle: copies the true labels; checks the build and the scorer.
harbor run -p $T -a oracle -e docker -o jobs/harbor

# Open a shell in the agent environment.
harbor task start-env -p $T -e docker -i

# An agent.
harbor run -p $T -a claude-code -m anthropic/claude-opus-5-5 -e docker -o jobs/harbor
harbor view jobs/harbor
```
