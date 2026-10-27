# Hard agent evaluation

This folder releases the two Hard agent conditions used in the paper:

| Condition | Agent workspace |
| --- | --- |
| `c2` (sandbox only) | `case/background.txt`, `case/questions.json`, `task.md` |
| `c3` (library available) | The same files, plus the complete 11-module `translation` runtime library and `MANUAL.md` |

The `payloads/` directory contains the **exact task and case files** from the
original evaluation for all 30 public Hard items. Its folders are named by the
public Hard item ID; `payload_manifest.json` records each item's original
question key and answer shape. Gold answers are in `../data/hard.json` and are
never copied into an agent workspace. The original comparison ran each item
three times per condition in a networked 2-CPU, 10-GB sandbox with a three-hour
wall-clock budget. Only `answer.json` was scored; optional `solution.py` was
retained for inspection.

`library/translation/` is the entire importable library that was available to
`c3`, including Z3, BDD, SAT, and Ganak backends. It was staged from the
evaluation snapshot with the original staging script and verified in a clean
Python interpreter. `MANUAL.md` is the generated manual that `c3` received.
The solver dependency versions in the local evaluation environment are recorded
in `requirements.txt`; install them in the container image used to run the
agent. Optional backends may require platform-specific wheels.

The workspace builder copies only the files listed in the table above.

## Prepare the two conditions

From the repository root:

```bash
python hard_agent/prepare_workspaces.py --output-root /tmp/hyperlogic-agent --rollouts 3
```

This creates `c2/<item-id>/r01..r03/` and `c3/<item-id>/r01..r03/`.
The program refuses to overwrite a nonempty output directory and checks that
each workspace contains only the files for its condition. **Run the agents in
separate containers**, mounting only one prepared workspace per container;
running them directly on a checkout can let an agent read the public gold file.

`run_agents.py` provides a Docker wrapper. Supply an image containing your
agent CLI and an agent command that reads `task.md`, for example:

```bash
python hard_agent/run_agents.py \
  --workspaces /tmp/hyperlogic-agent --condition c2 \
  --image YOUR_AGENT_IMAGE --env YOUR_API_KEY_VAR \
  --concurrency 4 -- YOUR_AGENT_COMMAND task.md

python hard_agent/run_agents.py \
  --workspaces /tmp/hyperlogic-agent --condition c3 \
  --image YOUR_AGENT_IMAGE --env YOUR_API_KEY_VAR \
  --concurrency 4 -- YOUR_AGENT_COMMAND task.md
```

The wrapper enforces the 2-CPU, 10-GB, three-hour defaults, stores a log and
status for each arm, and passes only the named environment variables. You must
choose the same model and reasoning setting for both conditions. Agent CLI,
provider route, sandbox image, and credentials are external dependencies; the
wrapper does not encode a private provider account or cloud service. The
original evaluation used Codex for one model and Claude Code for the others.

## Collect and score

Each agent writes `answer.json` at its workspace root, keyed by the qid in
`case/questions.json`, for example `{"q3": 42}`. The collector normalizes
on-shape JSON answers using `notation.py` (sets and rows are sorted, then
rendered in the reference-answer format) before semantic judging. Nested row
cells and grouped dictionaries are passed as JSON so their structure survives;
missing or invalid answer files count wrong and are not sent to the judge.

```bash
python hard_agent/collect_predictions.py \
  --workspaces /tmp/hyperlogic-agent --condition c2 --rollout 1 \
  --output /tmp/c2-r01.jsonl

python evaluate.py score --tier hard --predictions /tmp/c2-r01.jsonl \
  --judge-model YOUR_JUDGE_MODEL --judge-repeats 2 \
  --output /tmp/c2-r01-scores.json
```

Repeat for `c2` and `c3`, rollouts 1–3. `evaluate.py` uses one semantic judge
prompt and the same item-level scoring rule for both conditions. With two judge
calls, the first verdict determines the score and the agreement rate is
reported separately. For already judged responses, use its `--verdicts` option
to score offline. Aggregate one
condition's three score JSON files with:

```bash
python hard_agent/summarize_rollouts.py \
  /tmp/c2-r01-scores.json /tmp/c2-r02-scores.json /tmp/c2-r03-scores.json \
  --output /tmp/c2-summary.json
```

The published scorer evaluates the 30-item public subset. Its results should
be reported as public-subset results, separate from the paper's 100-item Hard
results. Different provider routes or container images can change agent
behavior even when the payloads, library, budget, and judge are identical.
