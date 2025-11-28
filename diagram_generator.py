import os
from typing import Dict, Any
from uuid import uuid4
from graphviz import Digraph

# Add Graphviz to PATH for Windows
os.environ["PATH"] += os.pathsep + r"C:\graphviz-14.0.4\bin"


def ensure_output_dir() -> str:
    """
    Ensure the 'static/diagrams' directory exists.
    Flask serves from 'static' by default.
    """
    out_dir = os.path.join("static", "diagrams")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ---- Color theme (modern pastel enterprise look) ----
COLORS = {
    "frontend":   "#d0e6ff",   # Light blue
    "gateway":    "#ffe2b5",   # Light orange
    "services":   "#e6ffd5",   # Light green
    "databases":  "#ffd5e6",   # Light pink
    "pipeline":   "#f2e5ff",   # Light purple
    "other":      "#e0e0e0",   # Grey
}

CLUSTER_BORDER = "#999999"   # Soft grey
NODE_BORDER = "#555555"      # Node border color


def generate_graphviz_diagram(arch_plan: Dict[str, Any]):
    """
    Take an architecture plan (components + connections) and render a
    Graphviz SVG + DOT source.

    Returns:
        (image_url: str, dot_source: str)
    """
    components = arch_plan.get("components", [])
    connections = arch_plan.get("connections", [])

    dot = Digraph(comment="Architecture Diagram")

    # ---- Global Graph Style ----
    # - rankdir=LR : left-to-right flow
    # - splines=ortho: right-angle edges
    # - nodesep / ranksep: give lines some breathing room
    dot.attr(
        "graph",
        rankdir="LR",
        bgcolor="#ffffff",
        splines="ortho",
        concentrate="true",
        nodesep="0.7",
        ranksep="1.2",
    )

    dot.attr(
        "node",
        shape="rounded",
        style="filled,rounded",
        penwidth="1.5",
        fontsize="12",
        fontname="Segoe UI",
        color=NODE_BORDER,
    )
    dot.attr(
        "edge",
        penwidth="1.2",
        fontsize="10",
        fontname="Segoe UI",
        color="#555555",
        arrowsize="0.8",
    )

    # ---- Layer grouping ----
    layers: Dict[str, list] = {
        "frontend": [],
        "gateway": [],
        "services": [],
        "databases": [],
        "pipeline": [],
        "other": [],
    }

    # Map component id -> layer name
    comp_layer: Dict[str, str] = {}

    for c in components:
        ctype = (c.get("type") or "").lower()

        if ctype in ("client", "web", "mobile", "frontend"):
            layer_name = "frontend"
        elif ctype in ("gateway", "api"):
            layer_name = "gateway"
        elif ctype in ("app", "service", "microservice"):
            layer_name = "services"
        elif ctype in ("database", "db", "storage"):
            layer_name = "databases"
        elif ctype in ("etl", "pipeline", "stream"):
            layer_name = "pipeline"
        else:
            layer_name = "other"

        layers[layer_name].append(c)
        if "id" in c:
            comp_layer[c["id"]] = layer_name

    # ---- Helper for drawing clusters ----
    def add_cluster(name: str, label: str, comps: list, color: str):
        """
        Draw a rounded cluster with a label and colored nodes.
        """
        if not comps:
            return

        with dot.subgraph(name=name) as sg:
            sg.attr(
                label=label,
                style="rounded",
                color=CLUSTER_BORDER,
                penwidth="1.5",
                bgcolor="#fafafa",
                fontsize="14",
                fontname="Segoe UI Semibold",
            )

            for c in comps:
                node_label = c.get("label", c.get("id", "Unknown"))
                sg.node(
                    c["id"],
                    node_label,
                    fillcolor=color,
                )

    # ---- Draw clusters ----
    add_cluster("cluster_frontend", "Frontend", layers["frontend"], COLORS["frontend"])
    add_cluster("cluster_gateway", "API Gateway", layers["gateway"], COLORS["gateway"])
    add_cluster("cluster_services", "Services", layers["services"], COLORS["services"])
    add_cluster("cluster_databases", "Databases", layers["databases"], COLORS["databases"])
    add_cluster("cluster_pipeline", "Data Pipeline", layers["pipeline"], COLORS["pipeline"])
    add_cluster("cluster_other", "Other", layers["other"], COLORS["other"])

    # ---- Enforce left-to-right layer order with invisible edges ----
    # This gives Graphviz strong hints about horizontal ordering and
    # significantly reduces line crossings.
    layer_order = ["frontend", "gateway", "services", "pipeline", "databases", "other"]
    prev_rep_node = None
    for lname in layer_order:
        comps = layers[lname]
        if not comps:
            continue
        rep_node = comps[0]["id"]
        if prev_rep_node:
            # Strong invisible edge to keep this layer to the right of previous
            dot.edge(prev_rep_node, rep_node, style="invis", weight="100")
        prev_rep_node = rep_node

    # Precompute a numeric index per layer for back-edge detection
    layer_index = {name: idx for idx, name in enumerate(layer_order, start=1)}

    # ---- Draw connections ----
    for conn in connections:
        src = conn.get("from")
        dst = conn.get("to")
        if not src or not dst:
            continue

        label = conn.get("label") or ""

        src_layer = comp_layer.get(src, "other")
        dst_layer = comp_layer.get(dst, "other")
        src_rank = layer_index.get(src_layer, 0)
        dst_rank = layer_index.get(dst_layer, 0)

        # Decide if this is a "back-edge" (right-to-left or cross-layer callback)
        is_back_edge = dst_rank < src_rank

        edge_kwargs: Dict[str, Any] = {}

        # Clean presentation: emphasize common protocol labels, but keep
        # most labels subtle.
        if label:
            edge_kwargs["label"] = label

        # Back-edges:
        # - don't influence ranking (constraint=false)
        # - drawn dashed so they visually read as callbacks / monitoring / etc.
        if is_back_edge:
            edge_kwargs["constraint"] = "false"
            edge_kwargs["style"] = "dashed"

        # Optionally, highlight very common protocols slightly darker
        if label.lower() in ("https", "http", "sql", "rest", "grpc"):
            edge_kwargs.setdefault("color", "#444444")

        dot.edge(src, dst, **edge_kwargs)

    # ---- Export SVG ----
    output_dir = ensure_output_dir()
    file_id = uuid4().hex
    filename = f"arch_{file_id}"
    filepath = os.path.join(output_dir, filename)

    dot.format = "svg"
    rendered_path = dot.render(filename=filepath, cleanup=True)

    # Convert filesystem path to Flask static URL
    return "/" + rendered_path.replace("\\", "/"), dot.source
