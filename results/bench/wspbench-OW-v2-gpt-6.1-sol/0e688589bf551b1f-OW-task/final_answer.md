{
  "before_program": {
    "op": "mux",
    "condition": {"op": "input", "index": 6},
    "then": {
      "op": "and",
      "left": {"op": "input", "index": 0},
      "right": {
        "op": "and",
        "left": {"op": "input", "index": 7},
        "right": {"op": "input", "index": 8}
      }
    },
    "else": {
      "op": "or",
      "left": {
        "op": "xor",
        "left": {
          "op": "nor",
          "left": {"op": "input", "index": 4},
          "right": {"op": "input", "index": 5}
        },
        "right": {
          "op": "or",
          "left": {"op": "input", "index": 7},
          "right": {"op": "input", "index": 9}
        }
      },
      "right": {
        "op": "nand",
        "left": {
          "op": "xor",
          "left": {"op": "input", "index": 0},
          "right": {"op": "input", "index": 4}
        },
        "right": {
          "op": "nand",
          "left": {"op": "input", "index": 3},
          "right": {"op": "input", "index": 8}
        }
      }
    }
  },
  "after_program": {
    "op": "mux",
    "condition": {"op": "input", "index": 6},
    "then": {
      "op": "and",
      "left": {"op": "input", "index": 0},
      "right": {
        "op": "and",
        "left": {"op": "input", "index": 7},
        "right": {"op": "input", "index": 8}
      }
    },
    "else": {
      "op": "and",
      "left": {
        "op": "xor",
        "left": {
          "op": "nor",
          "left": {"op": "input", "index": 4},
          "right": {"op": "input", "index": 5}
        },
        "right": {
          "op": "or",
          "left": {"op": "input", "index": 7},
          "right": {"op": "input", "index": 9}
        }
      },
      "right": {
        "op": "nand",
        "left": {
          "op": "xor",
          "left": {"op": "input", "index": 0},
          "right": {"op": "input", "index": 4}
        },
        "right": {
          "op": "nand",
          "left": {"op": "input", "index": 3},
          "right": {"op": "input", "index": 8}
        }
      }
    }
  }
}