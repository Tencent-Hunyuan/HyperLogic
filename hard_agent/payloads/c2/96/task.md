# Task

`case/background.txt` states a situation in Chinese and `case/questions.json`
asks one question about it. Answer it.

## What to write

Write your answer to `answer.json` in this directory, an object keyed by
question id:

```json
{"q1": <answer>}
```

Answer in Chinese, using the background's own terms for the things it names.
Each question declares the shape its answer takes:

- `q1`: a JSON integer

`answer.json` is the only thing scored.

If you write code, leave your final script as `solution.py`. It is not scored;
it is kept so that how an answer was reached is on the record.

## How you work

You have a shell, a filesystem and network access in this sandbox. Write and run
whatever code helps. Nothing about your approach is prescribed: solve the
questions however you judge best.
