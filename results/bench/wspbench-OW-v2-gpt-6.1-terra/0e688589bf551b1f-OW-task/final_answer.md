```json
{
  "before_program": {
    "op": "nand",
    "left": {"op": "input", "index": 0},
    "right": {"op": "not", "arg": {"op": "input", "index": 0}}
  },
  "after_program": {
    "op": "xor",
    "left": {"op": "input", "index": 0},
    "right": {"op": "input", "index": 0}
  }
}
```