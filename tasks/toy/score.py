"""Score the toy task: 1.0 if W is the identity."""
import json, os, sys
import struct

def read_npy(p):
    b = open(p, "rb").read()
    hl = struct.unpack("<H", b[8:10])[0]
    data = b[10 + hl:]
    return list(struct.unpack("<%df" % (len(data) // 4), data))

w = read_npy(os.path.join(os.environ["OPENWISH_FINAL_NET"], "W.npy"))
print(json.dumps({"ok": True, "score": float(w == [1.0, 0.0, 0.0, 1.0]), "W": w}))
