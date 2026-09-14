# -*- coding: utf-8 -*-
"""Schema-driven ComfyUI workflow builder.

Node input/output lists and widget ordering are derived from a live ComfyUI
/object_info dump, so slot indices and widgets_values can't silently drift from
the installed node packs.
"""
import json, os, uuid

# Frontend-only virtual nodes: not present in /object_info, declared by hand.
FRONTEND_ONLY = {
    "MarkdownNote": {"inputs": [], "outputs": [], "widgets": ["md"]},
    "Note":         {"inputs": [], "outputs": [], "widgets": ["text"]},
}

# Input types that are rendered as widgets rather than connection sockets.
# NB: /object_info emits combos in two shapes - a bare list of options, or the
# string "COMBO" with the options carried in the opts dict. Handle both.
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO",
                "COMFY_DYNAMICCOMBO_V3", "COMFY_MULTISELECT_V3"}

# Node types whose loader widget is followed by an upload-button widget.
UPLOAD_EXTRA = {
    "LoadImage": ["image"],
    "LoadAudio": [None, None],
    "LoadVideo": [],
}

# KJNodes dynamic-input nodes serialize a trailing "update inputs" button widget.
EXTRA_WIDGETS = {
    "ImageConcatMulti": [None],
    "JoinStringMulti": [None],
}

# Autogrow inputs are declared explicitly per node instance.
AUTOGROW = "COMFY_AUTOGROW_V3"


class Schema:
    def __init__(self, object_info_path):
        with open(object_info_path, "r", encoding="utf-8") as f:
            self.oi = json.load(f)

    def has(self, ntype):
        return ntype in self.oi or ntype in FRONTEND_ONLY

    def _ordered_inputs(self, ntype):
        """[(name, type, opts, section)] in required-then-optional order."""
        d = self.oi.get(ntype)
        if not d:
            return []
        out = []
        it = d.get("input", {})
        for sect in ("required", "optional"):
            for name, spec in (it.get(sect) or {}).items():
                t = spec[0]
                opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                out.append((name, t, opts, sect))
        return out

    def is_widget(self, t, opts):
        if isinstance(t, list):          # plain COMBO
            return True
        if opts.get("forceInput"):       # STRING/INT declared as a socket
            return False
        return t in WIDGET_TYPES

    def widget_names(self, ntype):
        """Widget slots in serialization order, including synthetic extras."""
        if ntype in FRONTEND_ONLY:
            return list(FRONTEND_ONLY[ntype]["widgets"])
        names = []
        for name, t, opts, _ in self._ordered_inputs(ntype):
            if t == AUTOGROW:
                continue
            if self.is_widget(t, opts):
                names.append(name)
                # ComfyUI appends a control_after_generate widget after seed ints
                if name in ("seed", "noise_seed") and t == "INT":
                    names.append("__control_after_generate__")
        if ntype in UPLOAD_EXTRA:
            names.extend("__upload_%d__" % i for i in range(len(UPLOAD_EXTRA[ntype])))
        if ntype in EXTRA_WIDGETS:
            names.extend("__btn_%d__" % i for i in range(len(EXTRA_WIDGETS[ntype])))
        return names

    def widget_defaults(self, ntype):
        vals = []
        if ntype in FRONTEND_ONLY:
            return [""] * len(FRONTEND_ONLY[ntype]["widgets"])
        for name, t, opts, _ in self._ordered_inputs(ntype):
            if t == AUTOGROW:
                continue
            if not self.is_widget(t, opts):
                continue
            if isinstance(t, list):
                vals.append(opts.get("default", t[0] if t else ""))
            elif t == "COMBO":
                o = opts.get("options") or []
                vals.append(opts.get("default", o[0] if o else ""))
            elif t == "COMFY_DYNAMICCOMBO_V3":
                o = opts.get("options") or []
                vals.append(opts.get("default", (o[0].get("key") if o and isinstance(o[0], dict) else "auto")))
            else:
                vals.append(opts.get("default", {"INT": 0, "FLOAT": 0.0,
                                                 "STRING": "", "BOOLEAN": False}.get(t, "")))
            if name in ("seed", "noise_seed") and t == "INT":
                vals.append("fixed")
        if ntype in UPLOAD_EXTRA:
            vals.extend(UPLOAD_EXTRA[ntype])
        if ntype in EXTRA_WIDGETS:
            vals.extend(EXTRA_WIDGETS[ntype])
        return vals

    def socket_inputs(self, ntype):
        """[(name, type)] for connection sockets, excluding autogrow."""
        if ntype in FRONTEND_ONLY:
            return []
        out = []
        for name, t, opts, _ in self._ordered_inputs(ntype):
            if t == AUTOGROW:
                continue
            if not self.is_widget(t, opts):
                out.append((name, t if isinstance(t, str) else "COMBO"))
        return out

    def widget_socket_type(self, ntype, wname):
        for name, t, opts, _ in self._ordered_inputs(ntype):
            if name == wname:
                return t if isinstance(t, str) else "COMBO"
        return "*"

    def outputs(self, ntype):
        if ntype in FRONTEND_ONLY:
            return []
        d = self.oi.get(ntype, {})
        return list(zip(d.get("output_name", []), d.get("output", [])))

    def combo_options(self, ntype, key):
        d = self.oi.get(ntype)
        if not d:
            return None
        it = d.get("input", {})
        for sect in ("required", "optional"):
            spec = (it.get(sect) or {}).get(key)
            if spec:
                t = spec[0]
                if isinstance(t, list):
                    return t
                o = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                if isinstance(o.get("options"), list):
                    return o["options"]
        return None


class Node:
    def __init__(self, wf, nid, ntype, pos, size, title=None, color=None,
                 bgcolor=None, mode=0, props=None, collapsed=False):
        self.wf = wf
        self.id = nid
        self.type = ntype
        self.pos = list(pos)
        self.size = list(size)
        self.title = title
        self.color = color
        self.bgcolor = bgcolor
        self.mode = mode
        self.collapsed = collapsed
        self.props = props or {}
        self.inputs = []          # [{name, type, link, widget?, label?, shape?}]
        self.outputs = []         # [{name, type, links[]}]
        self.widgets = []         # values
        self._wnames = []

    # -- construction -------------------------------------------------------
    def init_from_schema(self, sch, widget_overrides=None, autogrow=None,
                         manual_inputs=None, manual_outputs=None):
        self._wnames = sch.widget_names(self.type)
        self.widgets = sch.widget_defaults(self.type)
        for k, v in (widget_overrides or {}).items():
            if k not in self._wnames:
                raise KeyError("%s has no widget %r (has %s)" % (self.type, k, self._wnames))
            self.widgets[self._wnames.index(k)] = v
        if manual_inputs is not None:
            self.inputs = [dict(d) for d in manual_inputs]
        else:
            self.inputs = [{"name": n, "type": t, "link": None}
                           for n, t in sch.socket_inputs(self.type)]
        for entry in (autogrow or []):
            self.inputs.append({"label": entry["label"], "name": entry["name"],
                                "shape": 7, "type": entry["type"], "link": None})
        if manual_outputs is not None:
            self.outputs = [dict(d) for d in manual_outputs]
        else:
            self.outputs = [{"name": n, "type": t, "links": []}
                            for n, t in sch.outputs(self.type)]
        return self

    def promote_widget(self, sch, wname):
        """Turn a widget into a connectable socket (ComfyUI 'widget' input)."""
        if any(i["name"] == wname for i in self.inputs):
            return
        self.inputs.append({"name": wname,
                            "type": sch.widget_socket_type(self.type, wname),
                            "widget": {"name": wname}, "link": None})

    def islot(self, name):
        for i, inp in enumerate(self.inputs):
            if inp["name"] == name:
                return i
        raise KeyError("%s (#%s) has no input %r; has %s"
                       % (self.type, self.id, name, [i["name"] for i in self.inputs]))

    def oslot(self, name):
        for i, out in enumerate(self.outputs):
            if out["name"] == name:
                return i
        raise KeyError("%s (#%s) has no output %r; has %s"
                       % (self.type, self.id, name, [o["name"] for o in self.outputs]))

    def to_dict(self, order):
        d = {
            "id": self.id, "type": self.type,
            "pos": self.pos, "size": self.size,
            "flags": {"collapsed": True} if self.collapsed else {},
            "order": order, "mode": self.mode,
            "inputs": self.inputs, "outputs": self.outputs,
            "properties": dict(self.props),
            "widgets_values": self.widgets,
        }
        d["properties"].setdefault("Node name for S&R", self.type)
        if self.title:
            d["title"] = self.title
        if self.color:
            d["color"] = self.color
        if self.bgcolor:
            d["bgcolor"] = self.bgcolor
        return d


class Workflow:
    def __init__(self, sch):
        self.sch = sch
        self.nodes = []
        self.links = []
        self.groups = []
        self._nid = 0
        self._lid = 0

    def add(self, ntype, pos, size=(300, 100), **kw):
        if not self.sch.has(ntype):
            raise KeyError("node type not installed: %r" % ntype)
        self._nid += 1
        widget_overrides = kw.pop("widgets", None)
        autogrow = kw.pop("autogrow", None)
        manual_inputs = kw.pop("manual_inputs", None)
        manual_outputs = kw.pop("manual_outputs", None)
        n = Node(self, self._nid, ntype, pos, size, **kw)
        n.init_from_schema(self.sch, widget_overrides, autogrow,
                           manual_inputs, manual_outputs)
        self.nodes.append(n)
        return n

    def link(self, src, sout, dst, din):
        so = src.oslot(sout)
        di = dst.islot(din)
        ltype = src.outputs[so]["type"]
        self._lid += 1
        self.links.append([self._lid, src.id, so, dst.id, di, ltype])
        src.outputs[so]["links"].append(self._lid)
        dst.inputs[di]["link"] = self._lid
        return self._lid

    def wlink(self, src, sout, dst, wname):
        """Link into a widget slot, promoting it to a socket first."""
        dst.promote_widget(self.sch, wname)
        return self.link(src, sout, dst, wname)

    def group(self, title, nodes, color="#3f789e", pad=40, header=70):
        xs = [n.pos[0] for n in nodes]
        ys = [n.pos[1] for n in nodes]
        x2 = [n.pos[0] + n.size[0] for n in nodes]
        y2 = [n.pos[1] + n.size[1] for n in nodes]
        b = [min(xs) - pad, min(ys) - pad - header,
             max(x2) - min(xs) + pad * 2, max(y2) - min(ys) + pad * 2 + header]
        self.groups.append({"id": len(self.groups) + 1, "title": title,
                            "bounding": [round(v, 2) for v in b],
                            "color": color, "font_size": 24, "flags": {}})

    def to_dict(self):
        # topological-ish order: dependency depth
        by_id = {n.id: n for n in self.nodes}
        preds = {n.id: set() for n in self.nodes}
        for lid, oid, oslot, tid, tslot, lt in self.links:
            preds[tid].add(oid)
        order, depth = {}, {}

        def d(nid, stack=()):
            if nid in depth:
                return depth[nid]
            if nid in stack:
                return 0
            depth[nid] = 0 if not preds[nid] else 1 + max(
                (d(p, stack + (nid,)) for p in preds[nid]), default=0)
            return depth[nid]

        for n in self.nodes:
            d(n.id)
        for i, n in enumerate(sorted(self.nodes, key=lambda x: (depth[x.id], x.id))):
            order[n.id] = i
        return {
            "id": str(uuid.uuid4()),
            "revision": 0,
            "last_node_id": self._nid,
            "last_link_id": self._lid,
            "nodes": [n.to_dict(order[n.id]) for n in self.nodes],
            "links": self.links,
            "groups": self.groups,
            "config": {},
            "extra": {"ds": {"scale": 0.45, "offset": [0, 0]}},
            "version": 0.4,
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path
