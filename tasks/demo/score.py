import json, os, numpy as np
W = np.load(os.path.join(os.environ["OPENWISH_FINAL_NET"], "W.npy"))
n = np.load(os.path.join(os.environ["OPENWISH_FINAL_NET"], "norm.npy"))
print(json.dumps({"ok": True, "score": float(np.array_equal(W, np.eye(2)) and np.array_equal(n, [1, 1])), "W": W.ravel().tolist()}))
