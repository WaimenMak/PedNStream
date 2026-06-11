"""
Interactive network dashboard for saved PedNStream simulations.

Run with:
    python scripts/network_dashboard.py --name outputs/delft_paths --pos data/delft/node_positions.json
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import requests
from PIL import Image


MAP_METRICS = [
    "density",
    "link_flow",
    "speed",
    "num_pedestrians",
    "inflow",
    "outflow",
    "travel_time",
]
PANEL_METRICS = ["link_flow", "speed"]
DIRECTION_MODES = ["forward", "reverse", "mean", "max"]
DEFAULT_RANGES = {
    "density": (0.0, 8.0),
    "link_flow": (0.0, 3.0),
    "speed": (0.0, 1.5),
    "num_pedestrians": (0.0, 100.0),
    "inflow": (0.0, 40.0),
    "outflow": (0.0, 40.0),
    "travel_time": (0.0, 1000.0),
}


def reverse_link_id(link_id: str) -> str:
    """Return the reversed directed link id."""
    u, v = link_id.split("-", 1)
    return f"{v}-{u}"


def parse_node_path(raw_path: str | Iterable[int | str] | None) -> list[str]:
    """Parse a manual path like ``0,4,110,109`` into node id strings."""
    if raw_path is None:
        return []
    if isinstance(raw_path, str):
        text = raw_path.strip()
        if not text:
            return []
        for sep in ["->", "-", ";"]:
            text = text.replace(sep, ",")
        nodes = [part.strip() for part in text.split(",") if part.strip()]
    else:
        nodes = [str(part).strip() for part in raw_path if str(part).strip()]

    if len(nodes) < 2:
        raise ValueError("A path needs at least two node ids.")
    return nodes


def moving_average(values: np.ndarray | Iterable[float], window_size: int = 1) -> np.ndarray:
    """Return a centered-enough moving average using the notebook's valid window."""
    arr = np.asarray(values, dtype=float)
    window = max(1, int(window_size))
    if window <= 1 or len(arr) == 0:
        return arr.copy()
    if window > len(arr):
        return np.array([float(np.mean(arr))])
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr, kernel, mode="valid")


def _series(link_data: dict, link_id: str, metric: str) -> np.ndarray | None:
    payload = link_data.get(link_id)
    if payload is None or metric not in payload:
        return None
    return np.asarray(payload[metric], dtype=float)


def combine_directional_series(
    link_data: dict,
    u: str | int,
    v: str | int,
    metric: str,
    mode: str = "mean",
) -> np.ndarray | None:
    """
    Combine forward and reverse link series for one path segment.

    ``forward`` and ``reverse`` fall back to the available opposite direction so
    sparse one-way networks can still be analyzed.
    """
    if mode not in DIRECTION_MODES:
        raise ValueError(f"Invalid direction mode: {mode}")

    fwd_id = f"{u}-{v}"
    rev_id = f"{v}-{u}"
    fwd = _series(link_data, fwd_id, metric)
    rev = _series(link_data, rev_id, metric)

    if mode == "forward":
        return fwd if fwd is not None else rev
    if mode == "reverse":
        return rev if rev is not None else fwd

    if fwd is not None and rev is not None:
        common = min(len(fwd), len(rev))
        if mode == "mean":
            return 0.5 * (fwd[:common] + rev[:common])
        return np.maximum(fwd[:common], rev[:common])

    return fwd if fwd is not None else rev


def combine_map_series(
    link_data: dict,
    u: str | int,
    v: str | int,
    metric: str,
    mode: str = "mean",
) -> np.ndarray | None:
    """
    Combine directions for map/video display.

    In the original Folium dashboard, bidirectional density and pedestrian
    counts were displayed as combined corridor totals rather than averages.
    """
    if mode not in DIRECTION_MODES:
        raise ValueError(f"Invalid direction mode: {mode}")

    if mode in ("forward", "reverse", "max"):
        return combine_directional_series(link_data, u, v, metric, mode)

    fwd_id = f"{u}-{v}"
    rev_id = f"{v}-{u}"
    fwd = _series(link_data, fwd_id, metric)
    rev = _series(link_data, rev_id, metric)

    if fwd is not None and rev is not None:
        common = min(len(fwd), len(rev))
        if metric in {"density", "num_pedestrians", "link_flow", "inflow", "outflow"}:
            return fwd[:common] + rev[:common]
        if metric == "speed":
            return np.maximum(fwd[:common], rev[:common])
        return 0.5 * (fwd[:common] + rev[:common])

    return fwd if fwd is not None else rev


def compute_path_metric(
    link_data: dict,
    node_path: Iterable[int | str],
    metric: str = "link_flow",
    mode: str = "mean",
) -> np.ndarray:
    """Average a metric across all valid links in a node path for every time step."""
    nodes = parse_node_path(node_path)
    rows = []
    common_len = None

    for u, v in zip(nodes[:-1], nodes[1:]):
        values = combine_directional_series(link_data, u, v, metric, mode)
        if values is None:
            continue
        common_len = len(values) if common_len is None else min(common_len, len(values))
        rows.append(values)

    if not rows or common_len is None:
        raise ValueError(f"No valid links with metric '{metric}' found along the path.")

    matrix = np.vstack([row[:common_len] for row in rows])
    return np.mean(matrix, axis=0)


def compute_network_metric(
    link_data: dict,
    metric: str = "link_flow",
    mode: str = "mean",
) -> np.ndarray:
    """Average a metric across the whole network for every time step."""
    if mode not in DIRECTION_MODES:
        raise ValueError(f"Invalid direction mode: {mode}")

    rows = []
    common_len = None
    processed_pairs: set[tuple[str, str]] = set()

    for link_id in sorted(link_data):
        if metric not in link_data[link_id]:
            continue
        u, v = link_id.split("-", 1)

        if mode in ("mean", "max"):
            pair = tuple(sorted((u, v), key=str))
            if pair in processed_pairs:
                continue
            values = combine_directional_series(link_data, u, v, metric, mode)
            processed_pairs.add(pair)
        else:
            values = _series(link_data, link_id, metric)

        if values is None:
            continue
        common_len = len(values) if common_len is None else min(common_len, len(values))
        rows.append(values)

    if not rows or common_len is None:
        raise ValueError(f"No valid links with metric '{metric}' found in the network.")

    matrix = np.vstack([row[:common_len] for row in rows])
    return np.mean(matrix, axis=0)


def compute_path_matrix(
    link_data: dict,
    node_path: Iterable[int | str],
    metric: str = "travel_time",
    mode: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Build a link-by-time matrix and cumulative distance edges for a path."""
    nodes = parse_node_path(node_path)
    rows = []
    lengths = []
    common_len = None

    for u, v in zip(nodes[:-1], nodes[1:]):
        values = combine_directional_series(link_data, u, v, metric, mode)
        if values is None:
            continue

        fwd_id = f"{u}-{v}"
        rev_id = f"{v}-{u}"
        payload = link_data.get(fwd_id) or link_data.get(rev_id)
        length = float(payload.get("parameters", {}).get("length", 1.0))

        common_len = len(values) if common_len is None else min(common_len, len(values))
        rows.append(values)
        lengths.append(length)

    if not rows or common_len is None:
        raise ValueError(f"No valid links with metric '{metric}' found along the path.")

    matrix = np.vstack([row[:common_len] for row in rows])
    distance_edges = np.concatenate([[0.0], np.cumsum(lengths)])
    return matrix, distance_edges


def smoothed_time_axis(series_len: int, unit_time: float, window_size: int = 1) -> np.ndarray:
    """Time axis aligned with ``moving_average(..., mode='valid')``."""
    window = max(1, int(window_size))
    if window <= 1:
        return np.arange(series_len, dtype=float) * unit_time
    return np.arange(series_len, dtype=float) * unit_time + (window - 1) * unit_time / 2


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def interpolate_color(value: float, vmin: float, vmax: float, palette: list[str]) -> str:
    """Interpolate through a small hex palette."""
    if not math.isfinite(value):
        return "#a8adb4"
    if vmax <= vmin:
        return palette[-1]

    t = min(1.0, max(0.0, (value - vmin) / (vmax - vmin)))
    scaled = t * (len(palette) - 1)
    idx = min(len(palette) - 2, int(math.floor(scaled)))
    frac = scaled - idx
    c1 = _hex_to_rgb(palette[idx])
    c2 = _hex_to_rgb(palette[idx + 1])
    rgb = tuple(round(c1[channel] + (c2[channel] - c1[channel]) * frac) for channel in range(3))
    return _rgb_to_hex(rgb)


def metric_palette(metric: str) -> list[str]:
    if metric == "speed":
        return ["#ff0000", "#ffff00", "#008000"]
    return ["#008000", "#ffff00", "#ff0000"]


def metric_label(metric: str) -> str:
    labels = {
        "density": "Density",
        "link_flow": "Link flow",
        "speed": "Speed",
        "num_pedestrians": "Pedestrians",
        "inflow": "Inflow",
        "outflow": "Outflow",
        "travel_time": "Travel time",
    }
    return labels.get(metric, metric.replace("_", " ").title())


@dataclass
class SelectedPath:
    nodes: list[str]
    source: str
    error: str | None = None


class NetworkDashboard:
    """Dash-backed dashboard data model and app factory."""

    def __init__(self, data_path: str | os.PathLike, pos: dict, zoom_start: int = 14):
        self.data_path = Path(data_path)
        self.pos = {str(k): (float(v[0]), float(v[1])) for k, v in pos.items()}
        self.zoom_start = zoom_start

        with open(self.data_path / "link_data.json", "r") as f:
            self.link_data = json.load(f)
        with open(self.data_path / "network_params.json", "r") as f:
            self.network_params = json.load(f)

        self.controller_nodes = self._detect_controller_nodes()
        self.controller_links = self._detect_controller_links()
        self.unit_time = float(self.network_params.get("unit_time", 1.0))
        self.max_time = min(
            len(payload["density"])
            for payload in self.link_data.values()
            if "density" in payload
        )
        self.od_paths = self.network_params.get("od_paths", {}) or {}

        lats = [coords[1] for coords in self.pos.values()]
        lons = [coords[0] for coords in self.pos.values()]
        self.center = [(max(lats) + min(lats)) / 2, (max(lons) + min(lons)) / 2]
        self.bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
        self.path_link_ids: set[str] = set()

    def resolve_path(
        self,
        scope: str,
        od_pair: str | None,
        path_index: int | str | None,
        manual_path: str | None,
    ) -> SelectedPath:
        """Resolve the current UI path selection."""
        if scope != "path":
            return SelectedPath([], "Whole network")

        if manual_path and manual_path.strip():
            try:
                return SelectedPath(parse_node_path(manual_path), "Manual path")
            except ValueError as exc:
                return SelectedPath([], "Manual path", str(exc))

        if od_pair and od_pair in self.od_paths:
            paths = self.od_paths[od_pair]
            if not paths:
                return SelectedPath([], f"OD {od_pair}", "The selected OD pair has no paths.")
            try:
                idx = min(max(0, int(path_index or 0)), len(paths) - 1)
            except ValueError:
                idx = 0
            return SelectedPath(parse_node_path(paths[idx]), f"OD {od_pair}, path {idx}")

        return SelectedPath([], "Path", "Choose an OD path or enter a manual path.")

    def create_app(self):
        """Create the Dash application. Dashboard packages are imported lazily."""
        try:
            import dash
            import dash_leaflet as dl
            import plotly.graph_objects as go
            from dash import Input, Output, State, dcc, html
        except ImportError as exc:
            raise ImportError(
                "Dashboard dependencies are missing. Install them with "
                "`pip install -e .[dashboard]`."
            ) from exc

        app = dash.Dash(__name__)
        app.index_string = self._index_string()
        od_options = [
            {"label": f"{od} ({len(paths)} paths)", "value": od}
            for od, paths in sorted(self.od_paths.items())
        ]
        default_od = od_options[0]["value"] if od_options else None
        default_vmin, default_vmax = DEFAULT_RANGES["density"]

        app.layout = html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H2("PedNStream Network Dashboard"),
                                html.Div(str(self.data_path), className="subtitle"),
                            ],
                            className="title-block",
                        ),
                        html.Div(
                            [
                                html.Button("Play", id="play-button", className="primary-button"),
                                html.Div(id="time-label", className="time-label"),
                            ],
                            className="play-row",
                        ),
                    ],
                    className="topbar",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Map metric"),
                                dcc.Dropdown(
                                    id="metric",
                                    options=[{"label": metric_label(m), "value": m} for m in MAP_METRICS],
                                    value="density",
                                    clearable=False,
                                ),
                                html.Label("Direction mode"),
                                dcc.Dropdown(
                                    id="direction-mode",
                                    options=[{"label": m.title(), "value": m} for m in DIRECTION_MODES],
                                    value="mean",
                                    clearable=False,
                                ),
                                html.Label("Time step"),
                                dcc.Slider(
                                    id="time-step",
                                    min=0,
                                    max=self.max_time - 1,
                                    value=0,
                                    step=1,
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                                dcc.Interval(id="play-interval", interval=450, disabled=True),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label("Color min"),
                                                dcc.Input(id="color-min", type="number", value=default_vmin, step=0.1),
                                            ],
                                            className="compact-field",
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Color max"),
                                                dcc.Input(id="color-max", type="number", value=default_vmax, step=0.1),
                                            ],
                                            className="compact-field",
                                        ),
                                    ],
                                    className="control-grid",
                                ),
                                html.H3("Metric scope"),
                                dcc.RadioItems(
                                    id="scope",
                                    options=[
                                        {"label": "Path", "value": "path"},
                                        {"label": "Whole network", "value": "network"},
                                    ],
                                    value="path" if od_options else "network",
                                    inline=True,
                                    className="radio-row",
                                ),
                                html.Label("OD path"),
                                dcc.Dropdown(
                                    id="od-pair",
                                    options=od_options,
                                    value=default_od,
                                    clearable=True,
                                    placeholder="No OD paths found",
                                ),
                                html.Label("Path index"),
                                dcc.Dropdown(id="path-index", clearable=False),
                                html.Label("Manual path override"),
                                dcc.Input(
                                    id="manual-path",
                                    type="text",
                                    placeholder="0,4,110,109",
                                    debounce=True,
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label("Smoothing"),
                                                dcc.Input(
                                                    id="smooth-window",
                                                    type="number",
                                                    value=10,
                                                    min=1,
                                                    step=1,
                                                ),
                                            ],
                                            className="compact-field",
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Heatmap"),
                                                dcc.Dropdown(
                                                    id="heatmap-metric",
                                                    options=[
                                                        {"label": metric_label(m), "value": m}
                                                        for m in ["travel_time", "speed", "link_flow"]
                                                    ],
                                                    value="travel_time",
                                                    clearable=False,
                                                ),
                                            ],
                                            className="compact-field",
                                        ),
                                    ],
                                    className="control-grid",
                                ),
                                html.H3("Video export"),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label("Start"),
                                                dcc.Input(
                                                    id="video-start",
                                                    type="number",
                                                    value=0,
                                                    min=0,
                                                    step=1,
                                                ),
                                            ],
                                            className="compact-field",
                                        ),
                                        html.Div(
                                            [
                                                html.Label("End"),
                                                dcc.Input(
                                                    id="video-end",
                                                    type="number",
                                                    value=min(50, self.max_time - 1),
                                                    min=0,
                                                    step=1,
                                                ),
                                            ],
                                            className="compact-field",
                                        ),
                                    ],
                                    className="control-grid",
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label("FPS"),
                                                dcc.Input(id="video-fps", type="number", value=5, min=1, max=30, step=1),
                                            ],
                                            className="compact-field",
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Format"),
                                                dcc.Dropdown(
                                                    id="video-format",
                                                    options=[
                                                        {"label": "GIF", "value": "gif"},
                                                        {"label": "MP4", "value": "mp4"},
                                                    ],
                                                    value="gif",
                                                    clearable=False,
                                                ),
                                            ],
                                            className="compact-field",
                                        ),
                                    ],
                                    className="control-grid",
                                ),
                                dcc.Checklist(
                                    id="video-background",
                                    options=[{"label": "Street background", "value": "street"}],
                                    value=[],
                                    className="checkbox-row",
                                ),
                                html.Button("Download Video", id="video-button", className="secondary-button"),
                                dcc.Download(id="video-download"),
                                html.Div(id="video-status", className="video-status"),
                                html.Div(id="selection-status", className="selection-status"),
                            ],
                            className="sidebar",
                        ),
                        html.Div(
                            [
                                dl.Map(
                                    [
                                        dl.TileLayer(),
                                        dl.LayerGroup(id="link-layer"),
                                        dl.LayerGroup(self._node_components(dl), id="node-layer"),
                                        dl.LayerGroup(self._controller_components(dl), id="controller-layer"),
                                    ],
                                    id="map",
                                    center=self.center,
                                    zoom=self.zoom_start,
                                    bounds=self.bounds,
                                    style={"height": "calc(100vh - 98px)", "width": "100%"},
                                ),
                                html.Div(id="colorbar-overlay", className="colorbar-overlay"),
                            ],
                            className="map-panel",
                        ),
                        html.Div(
                            [
                                dcc.Graph(id="metric-chart", config={"displayModeBar": False}),
                                dcc.Graph(id="heatmap-chart", config={"displayModeBar": False}),
                            ],
                            className="chart-panel",
                        ),
                    ],
                    className="main-grid",
                ),
            ],
            className="app-shell",
        )

        @app.callback(
            Output("play-interval", "disabled"),
            Output("play-button", "children"),
            Input("play-button", "n_clicks"),
            State("play-interval", "disabled"),
            prevent_initial_call=True,
        )
        def toggle_play(_clicks, disabled):
            next_disabled = not disabled
            return next_disabled, "Play" if next_disabled else "Pause"

        @app.callback(
            Output("time-step", "value"),
            Input("play-interval", "n_intervals"),
            State("time-step", "value"),
            prevent_initial_call=True,
        )
        def advance_time(_n, current):
            return (int(current or 0) + 1) % self.max_time

        @app.callback(
            Output("path-index", "options"),
            Output("path-index", "value"),
            Input("od-pair", "value"),
        )
        def update_path_indices(od_pair):
            paths = self.od_paths.get(od_pair, []) if od_pair else []
            options = [{"label": f"Path {idx}", "value": idx} for idx in range(len(paths))]
            return options, 0 if options else None

        @app.callback(
            Output("color-min", "value"),
            Output("color-max", "value"),
            Input("metric", "value"),
        )
        def update_color_defaults(metric):
            return DEFAULT_RANGES.get(metric, (0.0, 1.0))

        @app.callback(
            Output("link-layer", "children"),
            Output("metric-chart", "figure"),
            Output("heatmap-chart", "figure"),
            Output("time-label", "children"),
            Output("selection-status", "children"),
            Output("colorbar-overlay", "children"),
            Input("time-step", "value"),
            Input("metric", "value"),
            Input("direction-mode", "value"),
            Input("scope", "value"),
            Input("od-pair", "value"),
            Input("path-index", "value"),
            Input("manual-path", "value"),
            Input("smooth-window", "value"),
            Input("heatmap-metric", "value"),
            Input("color-min", "value"),
            Input("color-max", "value"),
        )
        def update_dashboard(
            time_step,
            metric,
            mode,
            scope,
            od_pair,
            path_index,
            manual_path,
            smooth_window,
            heatmap_metric,
            color_min,
            color_max,
        ):
            selected = self.resolve_path(scope, od_pair, path_index, manual_path)
            path_links = self._path_link_ids(selected.nodes)
            links = self._link_components(
                dl,
                int(time_step or 0),
                metric,
                mode,
                color_min,
                color_max,
                path_links,
            )
            chart = self._metric_figure(go, selected, scope, mode, int(smooth_window or 1))
            heatmap = self._heatmap_figure(go, selected, scope, mode, heatmap_metric)

            time_seconds = int(time_step or 0) * self.unit_time
            time_text = f"Step {int(time_step or 0)} | {time_seconds:g} s"
            if selected.error:
                status = selected.error
            elif scope == "network":
                status = "Whole network mean over valid links."
            else:
                status = f"{selected.source}: {' -> '.join(selected.nodes)}"
            colorbar = self._colorbar_components(html, metric, color_min, color_max)
            return links, chart, heatmap, time_text, status, colorbar

        @app.callback(
            Output("video-download", "data"),
            Output("video-status", "children"),
            Input("video-button", "n_clicks"),
            State("video-start", "value"),
            State("video-end", "value"),
            State("video-fps", "value"),
            State("video-format", "value"),
            State("metric", "value"),
            State("direction-mode", "value"),
            State("scope", "value"),
            State("od-pair", "value"),
            State("path-index", "value"),
            State("manual-path", "value"),
            State("color-min", "value"),
            State("color-max", "value"),
            State("video-background", "value"),
            prevent_initial_call=True,
        )
        def download_video(
            _clicks,
            start,
            end,
            fps,
            video_format,
            metric,
            mode,
            scope,
            od_pair,
            path_index,
            manual_path,
            color_min,
            color_max,
            video_background,
        ):
            try:
                selected = self.resolve_path(scope, od_pair, path_index, manual_path)
                path_links = self._path_link_ids(selected.nodes)
                show_street_background = "street" in (video_background or [])
                payload = self.generate_video_bytes(
                    start_time=int(start or 0),
                    end_time=int(end if end is not None else min(50, self.max_time - 1)),
                    fps=int(fps or 5),
                    video_format=video_format or "gif",
                    metric=metric,
                    mode=mode,
                    color_min=color_min,
                    color_max=color_max,
                    path_links=path_links,
                    show_street_background=show_street_background,
                )
                return payload, f"Generated {payload['filename']}."
            except Exception as exc:
                return None, f"Video export failed: {exc}"

        return app

    def run_dashboard(self, host: str = "127.0.0.1", port: int = 8050, debug: bool = False):
        app = self.create_app()
        app.run(host=host, port=port, debug=debug)

    def generate_video_bytes(
        self,
        start_time: int,
        end_time: int,
        fps: int,
        video_format: str,
        metric: str,
        mode: str,
        color_min: float | None,
        color_max: float | None,
        path_links: set[str] | None = None,
        show_street_background: bool = False,
        width: int = 1100,
        height: int = 780,
    ) -> dict:
        """Render the network animation to downloadable GIF or MP4 bytes."""
        try:
            import imageio
        except ImportError as exc:
            raise RuntimeError("Install imageio or the project video extra to export video.") from exc

        start_time = max(0, min(int(start_time), self.max_time - 1))
        end_time = max(start_time, min(int(end_time), self.max_time - 1))
        fps = max(1, int(fps))
        video_format = video_format.lower()
        if video_format not in {"gif", "mp4"}:
            raise ValueError("Video format must be 'gif' or 'mp4'.")

        frame_indices = range(start_time, end_time + 1)
        frames = [
            self._render_video_frame(
                time_step=t,
                metric=metric,
                mode=mode,
                color_min=color_min,
                color_max=color_max,
                path_links=path_links or set(),
                show_street_background=show_street_background,
                width=width,
                height=height,
            )
            for t in frame_indices
        ]

        suffix = f".{video_format}"
        if video_format == "gif":
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                output_path = tmp.name
            try:
                imageio.mimsave(output_path, frames, format="GIF", duration=1.0 / fps)
                output_bytes = Path(output_path).read_bytes()
            finally:
                Path(output_path).unlink(missing_ok=True)
            mime = "image/gif"
        else:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                output_path = tmp.name
            try:
                imageio.mimsave(output_path, frames, format="mp4", fps=fps, macro_block_size=1)
                output_bytes = Path(output_path).read_bytes()
            except Exception as exc:
                raise RuntimeError(
                    "MP4 export requires an imageio ffmpeg backend. Try GIF, or install imageio-ffmpeg/ffmpeg."
                ) from exc
            finally:
                Path(output_path).unlink(missing_ok=True)
            mime = "video/mp4"

        filename = f"network_{metric}_{start_time}_{end_time}.{video_format}"
        return {
            "content": base64.b64encode(output_bytes).decode("ascii"),
            "filename": filename,
            "type": mime,
            "base64": True,
        }

    def _render_video_frame(
        self,
        time_step: int,
        metric: str,
        mode: str,
        color_min: float | None,
        color_max: float | None,
        path_links: set[str],
        show_street_background: bool,
        width: int,
        height: int,
    ) -> np.ndarray:
        os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
        os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        dpi = 100
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#eef2f5")
        self._set_video_bounds(ax)
        if show_street_background:
            self._draw_street_background(ax)

        vmin, vmax = DEFAULT_RANGES.get(metric, (0.0, 1.0))
        vmin = float(vmin if color_min is None else color_min)
        vmax = float(vmax if color_max is None else color_max)
        palette = metric_palette(metric)

        for link in self._rendered_links(time_step, metric, mode, vmin, vmax, palette, path_links):
            ax.plot(
                [link["start"][0], link["end"][0]],
                [link["start"][1], link["end"][1]],
                color=link["color"],
                linewidth=link["width"],
                alpha=link["opacity"],
                solid_capstyle="round",
                zorder=3 if link["in_path"] else 2,
            )

        self._draw_video_nodes(ax)
        self._format_video_axes(ax, metric, time_step)
        fig.tight_layout(pad=0.35)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return frame

    def _rendered_links(
        self,
        time_step: int,
        metric: str,
        mode: str,
        vmin: float,
        vmax: float,
        palette: list[str],
        path_links: set[str],
    ):
        processed_pairs: set[tuple[str, str]] = set()
        for link_id in sorted(self.link_data):
            u, v = link_id.split("-", 1)
            pair = tuple(sorted((u, v), key=str))
            if mode in ("mean", "max") and pair in processed_pairs:
                continue

            values = combine_map_series(self.link_data, u, v, metric, mode)
            if values is None or time_step >= len(values) or u not in self.pos or v not in self.pos:
                continue

            value = float(values[time_step])
            in_path = link_id in path_links or reverse_link_id(link_id) in path_links
            width = self._line_width(metric, value, vmax)
            if in_path:
                width = max(width, 9)
            yield {
                "start": self.pos[u],
                "end": self.pos[v],
                "color": "#101820" if in_path else interpolate_color(value, vmin, vmax, palette),
                "width": width,
                "opacity": 0.98 if in_path else 0.8,
                "in_path": in_path,
            }
            processed_pairs.add(pair)

    def _draw_video_nodes(self, ax):
        origin_nodes = {str(n) for n in self.network_params.get("origin_nodes", [])}
        destination_nodes = {str(n) for n in self.network_params.get("destination_nodes", [])}
        for node_id, (lon, lat) in self.pos.items():
            if node_id in origin_nodes and node_id in destination_nodes:
                color, size = "#8e44ad", 32
            elif node_id in origin_nodes:
                color, size = "#c0392b", 28
            elif node_id in destination_nodes:
                color, size = "#1f618d", 28
            else:
                color, size = "#34495e", 10
            ax.scatter(lon, lat, s=size, color=color, edgecolor="white", linewidth=0.6, zorder=4)
        for node_id in self.controller_nodes:
            if node_id not in self.pos:
                continue
            lon, lat = self.pos[node_id]
            ax.scatter(
                lon,
                lat,
                s=96,
                facecolors="none",
                edgecolors="#f59e0b",
                linewidths=2.2,
                zorder=5,
            )

    def _set_video_bounds(self, ax):
        lons = [coords[0] for coords in self.pos.values()]
        lats = [coords[1] for coords in self.pos.values()]
        lon_pad = max((max(lons) - min(lons)) * 0.04, 1e-6)
        lat_pad = max((max(lats) - min(lats)) * 0.04, 1e-6)
        ax.set_xlim(min(lons) - lon_pad, max(lons) + lon_pad)
        ax.set_ylim(min(lats) - lat_pad, max(lats) + lat_pad)

    def _format_video_axes(self, ax, metric: str, time_step: int):
        self._set_video_bounds(ax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        time_seconds = time_step * self.unit_time
        ax.set_title(
            f"{metric_label(metric)} | step {time_step} | {time_seconds:g} s",
            fontsize=14,
            color="#1f2933",
            pad=8,
        )

    def _draw_street_background(self, ax):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zoom = self._tile_zoom_for_bounds(xlim, ylim)
        x_min, y_max = self._lonlat_to_tile(xlim[0], ylim[0], zoom)
        x_max, y_min = self._lonlat_to_tile(xlim[1], ylim[1], zoom)

        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                tile = self._fetch_osm_tile(zoom, x, y)
                if tile is None:
                    continue
                west, south, east, north = self._tile_bounds(x, y, zoom)
                ax.imshow(
                    tile,
                    extent=[west, east, south, north],
                    origin="upper",
                    interpolation="bilinear",
                    zorder=0,
                )

    def _fetch_osm_tile(self, zoom: int, x: int, y: int) -> np.ndarray | None:
        cache_dir = Path("/private/tmp/pednstream_tiles") / str(zoom) / str(x)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{y}.png"
        if cache_path.exists():
            return np.asarray(Image.open(cache_path).convert("RGB"))

        url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
        try:
            response = requests.get(
                url,
                timeout=8,
                headers={"User-Agent": "PedNStream dashboard video export"},
            )
            response.raise_for_status()
            cache_path.write_bytes(response.content)
            return np.asarray(Image.open(cache_path).convert("RGB"))
        except Exception:
            cache_path.unlink(missing_ok=True)
            return None

    @staticmethod
    def _tile_zoom_for_bounds(xlim: tuple[float, float], ylim: tuple[float, float]) -> int:
        lon_span = max(abs(xlim[1] - xlim[0]), 1e-9)
        lat_span = max(abs(ylim[1] - ylim[0]), 1e-9)
        target_tiles = 4
        zoom_lon = math.log2(360.0 * target_tiles / lon_span)
        zoom_lat = math.log2(170.0 * target_tiles / lat_span)
        return int(min(18, max(1, math.floor(min(zoom_lon, zoom_lat)))))

    @staticmethod
    def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
        lat = min(85.05112878, max(-85.05112878, lat))
        n = 2**zoom
        x = int((lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return max(0, min(n - 1, x)), max(0, min(n - 1, y))

    @staticmethod
    def _tile_bounds(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
        n = 2**zoom
        west = x / n * 360.0 - 180.0
        east = (x + 1) / n * 360.0 - 180.0

        def tile_y_to_lat(tile_y: int) -> float:
            lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * tile_y / n)))
            return math.degrees(lat_rad)

        north = tile_y_to_lat(y)
        south = tile_y_to_lat(y + 1)
        return west, south, east, north

    def _node_components(self, dl):
        origin_nodes = {str(n) for n in self.network_params.get("origin_nodes", [])}
        destination_nodes = {str(n) for n in self.network_params.get("destination_nodes", [])}
        nodes = []
        for node_id, (lon, lat) in self.pos.items():
            if node_id in origin_nodes and node_id in destination_nodes:
                color = "#8e44ad"
                radius = 7
            elif node_id in origin_nodes:
                color = "#c0392b"
                radius = 6
            elif node_id in destination_nodes:
                color = "#1f618d"
                radius = 6
            else:
                color = "#34495e"
                radius = 3
            nodes.append(
                dl.CircleMarker(
                    center=[lat, lon],
                    radius=radius,
                    color="#ffffff",
                    weight=1,
                    fillColor=color,
                    fillOpacity=0.9,
                    children=[dl.Tooltip(f"Node {node_id}")],
                )
            )
        return nodes

    def _controller_components(self, dl):
        nodes = []
        for node_id in sorted(self.controller_nodes, key=lambda value: int(value) if value.isdigit() else value):
            if node_id not in self.pos:
                continue
            lon, lat = self.pos[node_id]
            connected = sorted(
                link_id
                for link_id in self.controller_links
                if link_id.startswith(f"{node_id}-") or link_id.endswith(f"-{node_id}")
            )
            tooltip = f"Controller node {node_id}"
            if connected:
                tooltip += f" | controlled links: {', '.join(connected[:6])}"
                if len(connected) > 6:
                    tooltip += f", +{len(connected) - 6} more"
            nodes.append(
                dl.CircleMarker(
                    center=[lat, lon],
                    radius=10,
                    color="#f59e0b",
                    weight=3,
                    fill=False,
                    opacity=1.0,
                    children=[dl.Tooltip(tooltip)],
                )
            )
        return nodes

    def _detect_controller_nodes(self) -> set[str]:
        nodes: set[str] = set()
        controller_config = self.network_params.get("controllers", {}) or {}

        for node_id in controller_config.get("nodes", []) or []:
            nodes.add(str(node_id))

        for link_id in controller_config.get("links", []) or []:
            link_id = str(link_id)
            if "-" in link_id:
                u, v = link_id.split("-", 1)
                nodes.update([u, v])

        for link_id, payload in self.link_data.items():
            has_saved_control = any(
                key in payload
                for key in ("front_gate_width", "back_gate_width", "separator_width")
            ) or bool(payload.get("is_separator"))
            if has_saved_control and "-" in link_id:
                u, v = link_id.split("-", 1)
                nodes.update([u, v])

        return nodes

    def _detect_controller_links(self) -> set[str]:
        links: set[str] = set()
        controller_config = self.network_params.get("controllers", {}) or {}
        links.update(str(link_id) for link_id in controller_config.get("links", []) or [])

        for link_id, payload in self.link_data.items():
            has_saved_control = any(
                key in payload
                for key in ("front_gate_width", "back_gate_width", "separator_width")
            ) or bool(payload.get("is_separator"))
            if has_saved_control:
                links.add(link_id)

        return links

    def _link_components(
        self,
        dl,
        time_step: int,
        metric: str,
        mode: str,
        color_min: float | None,
        color_max: float | None,
        path_links: set[str],
    ):
        vmin, vmax = DEFAULT_RANGES.get(metric, (0.0, 1.0))
        vmin = float(vmin if color_min is None else color_min)
        vmax = float(vmax if color_max is None else color_max)
        palette = metric_palette(metric)
        components = []
        processed_pairs: set[tuple[str, str]] = set()

        for link_id in sorted(self.link_data):
            u, v = link_id.split("-", 1)
            pair = tuple(sorted((u, v), key=str))
            if mode in ("mean", "max") and pair in processed_pairs:
                continue

            values = combine_map_series(self.link_data, u, v, metric, mode)
            if values is None or time_step >= len(values) or u not in self.pos or v not in self.pos:
                continue

            value = float(values[time_step])
            in_path = link_id in path_links or reverse_link_id(link_id) in path_links
            color = "#101820" if in_path else interpolate_color(value, vmin, vmax, palette)
            opacity = 0.98 if in_path else 0.8
            width = self._line_width(metric, value, vmax)
            if in_path:
                width = max(width, 9)
            start = self.pos[u]
            end = self.pos[v]

            tooltip = (
                f"Link {u}->{v}"
                f" | {metric_label(metric)}: {value:.3g}"
                f" | mode: {mode}"
            )
            components.append(
                dl.Polyline(
                    positions=[[start[1], start[0]], [end[1], end[0]]],
                    color=color,
                    weight=width,
                    opacity=opacity,
                    children=[dl.Tooltip(tooltip)],
                )
            )
            processed_pairs.add(pair)
        return components

    def _metric_figure(self, go, selected: SelectedPath, scope: str, mode: str, smooth_window: int):
        fig = go.Figure()
        try:
            for metric, color in [("link_flow", "#1f77b4"), ("speed", "#d35400")]:
                if scope == "network":
                    values = compute_network_metric(self.link_data, metric, mode)
                else:
                    if selected.error:
                        raise ValueError(selected.error)
                    values = compute_path_metric(self.link_data, selected.nodes, metric, mode)
                smoothed = moving_average(values, smooth_window)
                time_axis = smoothed_time_axis(len(smoothed), self.unit_time, smooth_window)
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=smoothed,
                        mode="lines",
                        name=metric_label(metric),
                        line={"color": color, "width": 2},
                    )
                )
            title = "Whole-network metrics" if scope == "network" else "Path metrics"
            fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Mean value")
        except ValueError as exc:
            fig.add_annotation(text=str(exc), x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            fig.update_layout(title="Metric view")

        fig.update_layout(
            margin={"l": 45, "r": 20, "t": 45, "b": 42},
            legend={"orientation": "h", "y": 1.08, "x": 0},
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
        )
        fig.update_xaxes(showgrid=True, gridcolor="#e8edf2")
        fig.update_yaxes(showgrid=True, gridcolor="#e8edf2")
        return fig

    def _heatmap_figure(self, go, selected: SelectedPath, scope: str, mode: str, metric: str):
        fig = go.Figure()
        if scope != "path":
            fig.add_annotation(
                text="Switch to Path scope for the time-space heatmap.",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
            )
            fig.update_layout(title="Path time-space heatmap")
            return fig

        try:
            if selected.error:
                raise ValueError(selected.error)
            matrix, distance_edges = compute_path_matrix(self.link_data, selected.nodes, metric, mode)
            y = np.arange(matrix.shape[1], dtype=float) * self.unit_time
            x = 0.5 * (distance_edges[:-1] + distance_edges[1:])
            colorscale = "RdYlGn" if metric == "speed" else "RdYlGn_r"
            fig.add_trace(
                go.Heatmap(
                    z=matrix.T,
                    x=x,
                    y=y,
                    colorscale=colorscale,
                    colorbar={"title": metric_label(metric)},
                )
            )
            fig.update_layout(
                title=f"{metric_label(metric)} along path",
                xaxis_title="Distance along path (m)",
                yaxis_title="Time (s)",
            )
        except ValueError as exc:
            fig.add_annotation(text=str(exc), x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            fig.update_layout(title="Path time-space heatmap")

        fig.update_layout(margin={"l": 45, "r": 20, "t": 45, "b": 42}, paper_bgcolor="#ffffff")
        return fig

    def _path_link_ids(self, nodes: list[str]) -> set[str]:
        return {f"{u}-{v}" for u, v in zip(nodes[:-1], nodes[1:])}

    @staticmethod
    def _colorbar_components(html, metric: str, color_min: float | None, color_max: float | None):
        default_min, default_max = DEFAULT_RANGES.get(metric, (0.0, 1.0))
        vmin = default_min if color_min is None else float(color_min)
        vmax = default_max if color_max is None else float(color_max)
        midpoint = 0.5 * (vmin + vmax)
        colors = metric_palette(metric)
        gradient = f"linear-gradient(90deg, {colors[0]} 0%, {colors[1]} 50%, {colors[2]} 100%)"
        max_color_name = "green" if metric == "speed" else "red"

        return html.Div(
            [
                html.Div(
                    [
                        html.Span(f"{metric_label(metric)} color scale"),
                        html.Span(f"max = {vmax:g} ({max_color_name})", className="colorbar-max"),
                    ],
                    className="colorbar-title",
                ),
                html.Div(className="colorbar-gradient", style={"background": gradient}),
                html.Div(
                    [
                        html.Span(f"{vmin:g}"),
                        html.Span(f"{midpoint:g}"),
                        html.Span(f"{vmax:g}"),
                    ],
                    className="colorbar-ticks",
                ),
            ]
        )

    @staticmethod
    def _line_width(metric: str, value: float, vmax: float) -> float:
        if not math.isfinite(value):
            return 1.0
        if metric == "num_pedestrians":
            return min(10.0, max(1.0, value * 0.5))
        if metric == "speed":
            return max(1.0, min(10.0, value * 12.0))
        if metric == "density":
            return max(1.0, min(20.0, value * 12.0))
        scale = vmax if vmax > 0 else max(value, 1.0)
        return max(1.0, min(12.0, value / scale * 12.0))

    @staticmethod
    def _index_string() -> str:
        return f"""
        <!DOCTYPE html>
        <html>
            <head>
                {{%metas%}}
                <title>PedNStream Network Dashboard</title>
                {{%favicon%}}
                {{%css%}}
                <style>
                {NetworkDashboard._css()}
                </style>
            </head>
            <body>
                {{%app_entry%}}
                <footer>
                    {{%config%}}
                    {{%scripts%}}
                    {{%renderer%}}
                </footer>
            </body>
        </html>
        """

    @staticmethod
    def _css() -> str:
        return """
        body { margin: 0; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f4f6f8; color: #1f2933; }
        .topbar { height: 72px; display: flex; align-items: center; justify-content: space-between; padding: 0 18px; background: #ffffff; border-bottom: 1px solid #dce3ea; }
        .title-block h2 { margin: 0; font-size: 22px; font-weight: 650; }
        .subtitle { margin-top: 4px; color: #607080; font-size: 13px; }
        .play-row { display: flex; align-items: center; gap: 14px; }
        .primary-button { height: 36px; min-width: 74px; border: 0; border-radius: 6px; background: #1f77b4; color: #fff; font-weight: 650; cursor: pointer; }
        .secondary-button { width: 100%; height: 36px; margin-top: 12px; border: 1px solid #1f77b4; border-radius: 6px; background: #ffffff; color: #1f77b4; font-weight: 650; cursor: pointer; }
        .time-label { min-width: 120px; font-weight: 650; color: #34495e; }
        .main-grid { display: grid; grid-template-columns: 280px minmax(420px, 1fr) 430px; height: calc(100vh - 72px); }
        .sidebar { overflow-y: auto; padding: 14px; border-right: 1px solid #dce3ea; background: #ffffff; }
        .map-panel { min-width: 0; position: relative; }
        .chart-panel { overflow-y: auto; padding: 12px; border-left: 1px solid #dce3ea; background: #f8fafc; }
        label { display: block; margin: 13px 0 6px; color: #52616f; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0; }
        h3 { margin: 18px 0 4px; font-size: 15px; }
        input { width: 100%; height: 34px; box-sizing: border-box; border: 1px solid #cad3dc; border-radius: 6px; padding: 0 8px; font-size: 14px; }
        .control-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .compact-field label { margin-top: 12px; }
        .checkbox-row { margin-top: 12px; color: #1f2933; font-size: 13px; }
        .checkbox-row label { display: inline-flex; align-items: center; gap: 6px; margin: 0; text-transform: none; color: #1f2933; font-size: 13px; font-weight: 500; }
        .checkbox-row input { width: auto; height: auto; }
        .radio-row label { margin: 4px 12px 4px 0; text-transform: none; color: #1f2933; font-size: 13px; font-weight: 500; }
        .video-status { margin-top: 8px; color: #52616f; font-size: 12px; line-height: 1.35; overflow-wrap: anywhere; }
        .selection-status { margin-top: 14px; padding: 10px; background: #eef3f7; border-radius: 6px; color: #394b59; font-size: 13px; line-height: 1.35; overflow-wrap: anywhere; }
        .colorbar-overlay { position: absolute; left: 14px; bottom: 18px; z-index: 900; width: 260px; padding: 10px 12px; border-radius: 6px; background: rgba(255, 255, 255, 0.92); border: 1px solid rgba(31, 41, 51, 0.18); box-shadow: 0 8px 24px rgba(31, 41, 51, 0.18); pointer-events: none; }
        .colorbar-title { display: flex; justify-content: space-between; gap: 10px; font-size: 12px; font-weight: 700; color: #1f2933; }
        .colorbar-max { color: #52616f; font-weight: 650; white-space: nowrap; }
        .colorbar-gradient { height: 12px; margin: 8px 0 5px; border-radius: 999px; border: 1px solid rgba(31, 41, 51, 0.22); }
        .colorbar-ticks { display: flex; justify-content: space-between; font-size: 11px; color: #52616f; font-variant-numeric: tabular-nums; }
        .leaflet-container { background: #eef2f5; }
        """


def run_visualization(data_path: str, pos: dict, host: str = "127.0.0.1", port: int = 8050, debug: bool = False):
    dashboard = NetworkDashboard(data_path, pos)
    dashboard.run_dashboard(host=host, port=port, debug=debug)


def _load_positions(path: str | os.PathLike) -> dict[str, tuple[float, float]]:
    with open(path, "r") as f:
        return {str(k): tuple(v) for k, v in json.load(f).items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Network Dashboard Visualization")
    parser.add_argument("--name", type=str, default="outputs/delft_paths", help="Simulation output folder")
    parser.add_argument("--pos", type=str, default="data/delft/node_positions.json", help="Path to node_positions.json")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Dashboard host")
    parser.add_argument("--port", type=int, default=8050, help="Dashboard port")
    parser.add_argument("--debug", action="store_true", help="Run Dash in debug mode")
    args = parser.parse_args()

    run_visualization(args.name, _load_positions(args.pos), host=args.host, port=args.port, debug=args.debug)
