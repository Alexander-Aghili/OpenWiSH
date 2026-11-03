import json, os, numpy as np
req = json.load(open(os.environ["OPENWISH_REQ"]))
W = np.load(os.path.join(os.environ["OPENWISH_NET"], "W.npy"))
y = (W @ np.array(req["x"], dtype=np.float32)).tolist()
open(os.path.join(os.environ["OPENWISH_OUT"], "y.json"), "w").write(json.dumps(y))
print(json.dumps({"ok": True, "x": req["x"], "y": [round(v, 4) for v in y]}))
