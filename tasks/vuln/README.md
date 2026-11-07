# Vulnerability detection: Juliet CWE-476 on Qwen3.5-0.8B

The agent edits MLP weights of Qwen3.5-0.8B so the model catches NULL pointer dereferences. The model reads one C program and answers YES or NO to one fixed question: `Can calling entry() reach a NULL pointer dereference on any execution path?` The answer is YES when the YES logit beats the NO logit.

## The hidden set

[NIST Juliet C/C++ test suite v1.3](https://samate.nist.gov/SARD/test-suites/112), CWE-476, single-file testcases. Each testcase gives one matched pair. The vulnerable program keeps the `bad()` chain. The safe program keeps the first good variant the testcase's `good()` wrapper calls. `scripts/prepare_vuln.py` strips comments and `main()`, renames the entry function to `entry()`, and renames every helper and global by role (sink, source, flag ...). Names such as `bad`, `goodG2B` or `CWE476_...` never reach the model. Strings naming the label become neutral, and so does Juliet's dead-branch string. A seed picks the role names, and each pair appears under 3 seeds.

186 pairs, 1116 programs, 558 vulnerable. The 18 `null_check_after_deref` testcases are left out: both of their versions dereference an unchecked `malloc` result, so the Juliet label does not answer the question.

## Score

Recall on vulnerable programs, if the edited model answers YES on no more safe programs than the base model (407 of 558). Otherwise 0. The base model answers YES on 407 vulnerable and 407 safe programs: 50.0 % accuracy, score 0.729.

For reference, a one-line text rule ("YES unless the code compares a pointer with NULL") catches 432 and flags 108 safe programs: 79.0 % accuracy. Report accuracy and false positives next to the score.

## Setup

```bash
mkdir -p assets/vuln/download && cd assets/vuln/download
curl -LO https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip
mv 2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip juliet-1.3.zip
unzip -l juliet-1.3.zip | awk '{print $4}' | grep CWE476 | grep -E "__[a-z0-9_]+_[0-9]{2}\.c$" > single.txt
mkdir -p cwe476 && unzip -q -j juliet-1.3.zip $(cat single.txt) -d cwe476
cd ../../.. && assets/venvs/qwen/bin/python -I scripts/prepare_vuln.py
```

The model is the text-only Qwen3.5-0.8B export from `scripts/prepare_exp8.py` (`assets/exp8/qwen35-text`).

## Running inference on a GPU host

Set `OPENWISH_REMOTE` to a JSON file naming an ssh host with the same repo paths (see `python/adapters/remote.py`). Probes and the scorer then run there with CUDA. The agent, the kernel and the edit gate stay local.
