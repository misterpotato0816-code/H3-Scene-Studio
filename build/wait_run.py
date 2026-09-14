# -*- coding: utf-8 -*-
"""Poll a ComfyUI prompt to completion and report per-node timings + outputs.

Usage: python wait_run.py <prompt_id> [port] [timeout_s]
"""
import sys, io, json, time, urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

pid = sys.argv[1]
port = sys.argv[2] if len(sys.argv) > 2 else "8399"
timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 3600
BASE = "http://127.0.0.1:%s" % port


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.load(r)


t0 = time.time()
last_note = 0
while time.time() - t0 < timeout:
    try:
        h = get("/history/" + pid)
    except Exception as e:
        print("poll error:", e)
        time.sleep(5)
        continue

    if pid in h:
        entry = h[pid]
        status = entry.get("status", {})
        print("=" * 70)
        print("completed=%s  status=%s  wall=%.1fs"
              % (status.get("completed"), status.get("status_str"), time.time() - t0))

        # messages carry execution_start / execution_success / execution_error
        for m in status.get("messages", []):
            kind = m[0]
            data = m[1] if len(m) > 1 else {}
            if kind in ("execution_error", "execution_interrupted"):
                print("!! %s" % kind)
                for k in ("node_id", "node_type", "exception_type",
                          "exception_message"):
                    if k in data:
                        print("   %-18s %s" % (k, data[k]))
                tb = data.get("traceback")
                if tb:
                    print("   traceback tail:")
                    for line in (tb[-12:] if isinstance(tb, list) else str(tb).splitlines()[-12:]):
                        print("     ", line)

        outs = entry.get("outputs", {})
        print("--- outputs (%d nodes) ---" % len(outs))
        for nid, o in outs.items():
            keys = list(o.keys())
            print("  node %s : %s" % (nid, keys))
            for k in ("images", "video", "audio", "gifs"):
                for item in (o.get(k) or []):
                    if isinstance(item, dict):
                        print("      %s -> %s/%s" % (k, item.get("subfolder", ""), item.get("filename")))
            if "text" in o:
                t = o["text"]
                t = t[0] if isinstance(t, list) and t else t
                s = str(t)
                print("      text (%d chars):" % len(s))
                for line in s.splitlines()[:200]:
                    print("        ", line)
        sys.exit(0)

    # still running - show queue position occasionally
    if time.time() - last_note > 60:
        last_note = time.time()
        try:
            q = get("/queue")
            print("... running=%d pending=%d  elapsed=%.0fs"
                  % (len(q.get("queue_running", [])), len(q.get("queue_pending", [])),
                     time.time() - t0))
        except Exception:
            pass
    time.sleep(3)

print("TIMEOUT after %.0fs" % (time.time() - t0))
sys.exit(2)
