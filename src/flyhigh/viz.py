"""Annotated video of a running simulation: the room, what the brain sees, what the brain does.

`AnnotatedVideo` is an `on_tick` callback for `flyhigh.sim.Simulation` (M3 bodies) and
`flyhigh.flybody.sim.FlySimulation` (M4 flies) alike. Each frame carries four panels:

    overview camera        |  the agent's panorama, as the eyes sample it
    giant-fiber rate       |  the motor command, with escapes marked

so that a reaction can be read off the video instead of inferred from a log.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def mujoco_handles(sim):
    """(model, data) for either simulation class."""
    if hasattr(sim, "model"):
        return sim.model, sim.data
    return sim.physics.model.ptr, sim.physics.data.ptr


class AnnotatedVideo:
    def __init__(self, sim, path, agent: int = 0, fps: int = 20, every: int = 2,
                 width: int = 960, window_s: float = 3.0, title: str = ""):
        """`every`: render one frame per `every` ticks; `window_s`: the traces' rolling window."""
        import matplotlib
        matplotlib.use("Agg")
        import imageio.v2 as imageio
        import matplotlib.pyplot as plt
        import mujoco

        self.sim, self.agent, self.every, self.window_s, self.title = sim, agent, every, window_s, title
        model, _ = mujoco_handles(sim)
        self.overview = mujoco.Renderer(model, 360, 640)
        self.writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                                         macro_block_size=8)
        self.plt = plt
        self.fig = plt.figure(figsize=(width / 100, width * 0.62 / 100), dpi=100)
        self.fig.patch.set_facecolor("#111417")
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.25, 1], width_ratios=[1, 1],
                                   left=0.045, right=0.985, top=0.9, bottom=0.09, hspace=0.32, wspace=0.16)
        self.ax_room = self.fig.add_subplot(gs[0, 0])
        self.ax_eye = self.fig.add_subplot(gs[0, 1])
        self.ax_gf = self.fig.add_subplot(gs[1, 0])
        self.ax_cmd = self.fig.add_subplot(gs[1, 1])
        for ax in (self.ax_room, self.ax_eye):
            ax.set_xticks([]); ax.set_yticks([])
        for ax in (self.ax_gf, self.ax_cmd):
            ax.set_facecolor("#191d20")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color("#3a4247")
            ax.tick_params(colors="#98a2a8", labelsize=8)
            ax.grid(True, color="#262c30", linewidth=0.6)

    # ------------------------------------------------------------------ frame
    def _panel_titles(self):
        for ax, text in ((self.ax_room, "the room"), (self.ax_eye, "what the brain sees (az × el)"),
                         (self.ax_gf, "giant fiber DNp01 (Hz)"), (self.ax_cmd, "motor command")):
            ax.set_title(text, color="#e9edea", fontsize=9, loc="left", pad=6)

    def __call__(self, sim) -> None:
        if sim.tick % self.every:
            return
        data = mujoco_handles(sim)[1]
        self.overview.update_scene(data, camera="overview")
        room = self.overview.render()
        pano = sim.last_frames[self.agent].lum if sim.last_frames else np.full((180, 360), 0.5, np.float32)
        log = pl.DataFrame(sim.rows).filter(pl.col("agent") == self.agent) if sim.rows else None

        self.ax_room.clear(); self.ax_eye.clear(); self.ax_gf.clear(); self.ax_cmd.clear()
        self.ax_room.imshow(room); self.ax_room.set_xticks([]); self.ax_room.set_yticks([])
        self.ax_eye.imshow(pano, cmap="gray", vmin=0, vmax=1, extent=(-180, 180, -90, 90), aspect="auto")
        self.ax_eye.set_xticks([-180, -90, 0, 90, 180]); self.ax_eye.set_yticks([-90, 0, 90])
        self.ax_eye.tick_params(colors="#98a2a8", labelsize=8)

        if log is not None and log.height:
            t = log["t_ms"].to_numpy() / 1000.0
            lo = max(0.0, t[-1] - self.window_s)
            keep = t >= lo
            t_k = t[keep]
            gf = (log["gf_hz"].to_numpy()[keep] if "gf_hz" in log.columns else np.zeros_like(t_k))
            esc = log["escape"].to_numpy()[keep]
            self.ax_gf.plot(t_k, gf, color="#dc8071", linewidth=1.6)
            self.ax_gf.axhline(50, color="#98a2a8", linewidth=0.8, linestyle="--")
            self.ax_gf.set_ylim(-5, max(120.0, float(gf.max()) * 1.15 if gf.size else 120.0))
            self.ax_gf.set_xlabel("time (s)", color="#98a2a8", fontsize=8)
            for ax in (self.ax_gf, self.ax_cmd):
                ax.set_xlim(lo, lo + self.window_s)
                for k in np.nonzero(esc)[0]:
                    ax.axvspan(t_k[k] - 0.005, t_k[k] + 0.005, color="#dc8071", alpha=0.35, linewidth=0)
            for name, colour in (("forward", "#6cb0d8"), ("yaw", "#64b487"), ("lift", "#d6a44a")):
                self.ax_cmd.plot(t_k, log[name].to_numpy()[keep], color=colour, linewidth=1.5, label=name)
            self.ax_cmd.set_ylim(-1.05, 1.05)
            self.ax_cmd.legend(loc="upper left", fontsize=7, ncol=3, frameon=False, labelcolor="#98a2a8")
            self.ax_cmd.set_xlabel("time (s)", color="#98a2a8", fontsize=8)
            # the flag lasts one tick, the manoeuvre 100 ms: label the manoeuvre
            hot = log["escaping"].to_numpy()[keep] if "escaping" in log.columns else esc
            if hot.size and bool(hot[-1]):
                self.ax_eye.text(0, 68, "ESCAPE", color="#dc8071", fontsize=16, ha="center",
                                 fontweight="bold", family="monospace")

        self._panel_titles()
        head = self.title or ""
        if log is not None and log.height:
            head = f"{head}    t = {log['t_ms'][-1] / 1000:5.2f} s"
        self.fig.suptitle(head, color="#e9edea", fontsize=11, x=0.045, ha="left", family="monospace")
        self.fig.canvas.draw()
        frame = np.asarray(self.fig.canvas.buffer_rgba())[..., :3]
        self.writer.append_data(frame)

    def close(self) -> None:
        self.writer.close()
        self.overview.close()
        self.plt.close(self.fig)
