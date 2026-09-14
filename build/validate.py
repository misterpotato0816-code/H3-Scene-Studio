# -*- coding: utf-8 -*-
"""Static validation of a generated ComfyUI workflow against /object_info.

Checks: node types exist, link endpoints resolve, slot indices are consistent,
widgets_values arity matches the schema, combo values are selectable, and every
required connection input is satisfied.

Run: python validate.py <object_info.json> <workflow.json> [...]
"""
import sys, os, io, json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wfbuild import Schema, FRONTEND_ONLY, AUTOGROW   # noqa: E402

BYPASS_MODES = (2, 4)


def validate(sch, path):
    wf = json.load(open(path, "r", encoding="utf-8"))
    errs, warns = [], []
    nodes = {n["id"]: n for n in wf["nodes"]}

    for n in wf["nodes"]:
        t, nid = n["type"], n["id"]
        tag = "#%s %s" % (nid, t)
        if not sch.has(t):
            errs.append("%s : node type not installed" % tag)
            continue
        if t in FRONTEND_ONLY:
            continue

        # --- widgets arity -------------------------------------------------
        exp = sch.widget_names(t)
        got = n.get("widgets_values", [])
        if len(got) != len(exp):
            errs.append("%s : widgets_values has %d entries, schema expects %d %s"
                        % (tag, len(got), len(exp), exp))

        # --- combo values selectable --------------------------------------
        for i, wname in enumerate(exp):
            if wname.startswith("__") or i >= len(got):
                continue
            opts = sch.combo_options(t, wname)
            if opts is None or not opts:
                continue
            if isinstance(opts[0], dict):     # dynamic combo
                opts = [o.get("key") for o in opts]
            if got[i] not in opts:
                warns.append("%s : widget %r = %r not in options (%d available)"
                             % (tag, wname, got[i], len(opts)))

        # --- declared inputs must exist on the node ------------------------
        schema_in = {nm for nm, _ in sch.socket_inputs(t)}
        schema_w = set(exp)
        has_autogrow = any(
            spec[0] == AUTOGROW
            for sect in ("required", "optional")
            for spec in (sch.oi[t].get("input", {}).get(sect) or {}).values())
        for inp in n.get("inputs", []):
            nm = inp["name"]
            if nm in schema_in or nm in schema_w:
                continue
            if "." in nm and has_autogrow:
                continue                      # autogrow slot
            if t == "Any Switch (rgthree)" and nm.startswith("any_"):
                continue
            if nm.startswith("string_") or nm.startswith("image_"):
                continue                      # kjnodes dynamic inputs
            errs.append("%s : declared input %r not in schema" % (tag, nm))

        # --- required connection inputs satisfied --------------------------
        if n.get("mode", 0) not in BYPASS_MODES:
            req = (sch.oi[t].get("input", {}).get("required") or {})
            for nm, spec in req.items():
                ty = spec[0]
                o = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                if ty == AUTOGROW or sch.is_widget(ty, o):
                    continue
                found = [i for i in n.get("inputs", []) if i["name"] == nm]
                if not found:
                    errs.append("%s : missing required input socket %r" % (tag, nm))
                elif found[0].get("link") is None:
                    errs.append("%s : required input %r is not connected" % (tag, nm))

    # --- links -------------------------------------------------------------
    seen = set()
    for l in wf["links"]:
        lid, oid, oslot, tid, tslot, ltype = l
        if lid in seen:
            errs.append("link %s : duplicate id" % lid)
        seen.add(lid)
        if oid not in nodes:
            errs.append("link %s : origin node %s missing" % (lid, oid))
            continue
        if tid not in nodes:
            errs.append("link %s : target node %s missing" % (lid, tid))
            continue
        src, dst = nodes[oid], nodes[tid]
        if oslot >= len(src.get("outputs", [])):
            errs.append("link %s : origin slot %s out of range on #%s %s"
                        % (lid, oslot, oid, src["type"]))
        elif lid not in (src["outputs"][oslot].get("links") or []):
            errs.append("link %s : not back-referenced in #%s outputs[%s].links"
                        % (lid, oid, oslot))
        if tslot >= len(dst.get("inputs", [])):
            errs.append("link %s : target slot %s out of range on #%s %s"
                        % (lid, tslot, tid, dst["type"]))
        elif dst["inputs"][tslot].get("link") != lid:
            errs.append("link %s : target #%s inputs[%s].link = %r"
                        % (lid, tid, tslot, dst["inputs"][tslot].get("link")))

    # --- dangling input refs ----------------------------------------------
    linkids = {l[0] for l in wf["links"]}
    for n in wf["nodes"]:
        for i, inp in enumerate(n.get("inputs", [])):
            if inp.get("link") is not None and inp["link"] not in linkids:
                errs.append("#%s %s inputs[%d] refers to unknown link %s"
                            % (n["id"], n["type"], i, inp["link"]))
        for i, out in enumerate(n.get("outputs", [])):
            for lk in (out.get("links") or []):
                if lk not in linkids:
                    errs.append("#%s %s outputs[%d] refers to unknown link %s"
                                % (n["id"], n["type"], i, lk))
    return errs, warns


if __name__ == "__main__":
    oi = sys.argv[1]
    sch = Schema(oi)
    bad = 0
    for p in sys.argv[2:]:
        errs, warns = validate(sch, p)
        print("=" * 78)
        print("%s : %d error(s), %d warning(s)" % (os.path.basename(p), len(errs), len(warns)))
        for e in errs:
            print("  ERROR  %s" % e)
        for w in warns:
            print("  warn   %s" % w)
        bad += len(errs)
    sys.exit(1 if bad else 0)
