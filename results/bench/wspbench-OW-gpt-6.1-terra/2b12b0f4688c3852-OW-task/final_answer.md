```json
{
  "before_program": {
    "op": "not",
    "arg": {
      "op": "xor",
      "left": {"op": "input", "index": 3},
      "right": {"op": "input", "index": 4}
    }
  },
  "after_program": {
    "op": "not",
    "arg": {
      "op": "xor",
      "left": {"op": "input", "index": 3},
      "right": {"op": "input", "index": 4}
    }
  }
}
```