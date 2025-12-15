import os
import shutil
import subprocess
import math
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg


@dataclass
class CarSnap:
    soc: float
    desired_soc: float
    capacity: float
    urgency: float
    # Traveling-only fields (0 if not traveling)
    max_travel_time: float = 0.0
    travel_time_remaining: float = 0.0


@dataclass
class StationSnap:
    station_id: int
    nb_chargers: int
    max_nb_cars_waiting: int
    cars_charging: List[CarSnap]
    cars_waiting: List[CarSnap]
    cars_traveling: List[CarSnap]


@dataclass
class StepSnap:
    step_count: int
    reward: float
    action: int
    steps_until_next_arrival: int
    incoming: Optional[CarSnap]
    stations: List[StationSnap]


class EvVizRecorder:
    """
    Usage:
        rec = EvVizRecorder(env, output_path="videos/rollout.mp4", fps=2, snapshot_every=1)
        ...
        rec.record_step(action, reward, done)
        ...
        rec.save()
    """

    def __init__(
        self,
        env,
        output_path: str = "videos/ev_rollout.mp4",
        fps: int = 2,
        snapshot_every: int = 1,  # store one snapshot every N env steps
        snapshot_on_request_change: bool = False,  # also store if request appears/disappears
        snapshot_on_done: bool = True,  # always store final step
        convert_with_ffmpeg: bool = True,  # H.264/yuv420p conversion for browser playback
    ):
        self.env = env
        self.output_path = output_path
        self.fps = int(fps)

        self.snapshot_every = int(snapshot_every)
        self.snapshot_on_request_change = bool(snapshot_on_request_change)
        self.snapshot_on_done = bool(snapshot_on_done)
        self.convert_with_ffmpeg = bool(convert_with_ffmpeg)

        # Rendering settings
        self.width = 1920
        self.height = 1080
        self.dpi = 100

        # Color Palette (Normalized RGB)
        self.c_bg = (0.12, 0.12, 0.14)
        self.c_station_bg = (0.18, 0.18, 0.22)
        self.c_text_main = (0.9, 0.9, 0.9)
        self.c_text_sub = (0.6, 0.6, 0.6)

        self.c_road = (0.3, 0.3, 0.35)
        self.c_slot_empty = (0.25, 0.25, 0.28)
        self.c_slot_outline = (0.4, 0.4, 0.45)

        # Car State Colors
        self.c_car_neutral = (0.7, 0.7, 0.75)  # Gray
        self.c_car_travel = (0.95, 0.75, 0.2)  # Yellow
        self.c_car_wait = (0.85, 0.3, 0.3)  # Red
        self.c_car_charge = (0.2, 0.6, 0.9)  # Blue
        self.c_car_done = (0.3, 0.8, 0.4)  # Green

        self.nb_stations = len(env.stations)

        # Dynamic height calculation with padding
        self.margin_y = 60
        self.available_h = self.height - (self.margin_y * 2)
        self.station_h = min(180, self.available_h / max(self.nb_stations, 1) - 20)

        self.pos_incoming_x = self.width * 0.10
        self.pos_agent_x = self.width * 0.30
        self.pos_station_start_x = self.width * 0.60
        self.pos_station_w = self.width * 0.35

        # Snapshot storage
        self.snaps: List[StepSnap] = []
        self._last_request_ptr: Optional[int] = None

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

    # -------------------------
    # Snapshot logic
    # -------------------------

    def _tensor_to_scalar(self, val):
        if isinstance(val, torch.Tensor):
            return val.item()
        return val

    def _snap_car(self, car, traveling: bool) -> CarSnap:
        snap = CarSnap(
            soc=float(self._tensor_to_scalar(car.soc)),
            desired_soc=float(self._tensor_to_scalar(car.desired_soc)),
            capacity=float(self._tensor_to_scalar(car.capacity)),
            urgency=float(self._tensor_to_scalar(car.urgency)),
        )
        if traveling:
            snap.max_travel_time = float(self._tensor_to_scalar(car.max_travel_time))
            snap.travel_time_remaining = float(
                self._tensor_to_scalar(car.travel_time_remaining)
            )
        return snap

    def _should_snapshot(self, done: bool) -> bool:
        step = int(getattr(self.env, "step_count", len(self.snaps)))

        # Always snapshot final step if requested
        if done and self.snapshot_on_done:
            return True

        # Snapshot by frequency
        if self.snapshot_every > 1 and (step % self.snapshot_every) != 0:
            freq_ok = False
        else:
            freq_ok = True

        if not self.snapshot_on_request_change:
            return freq_ok

        # Snapshot on request appearance/disappearance
        cur = getattr(self.env, "car_to_route", None)
        cur_ptr = None if cur is None else id(cur)
        changed = cur_ptr != self._last_request_ptr
        self._last_request_ptr = cur_ptr

        return freq_ok or changed

    def record_step(self, action, reward, done):
        """
        Store a lightweight snapshot. No rendering is performed here.
        """
        if not self._should_snapshot(done):
            return

        a = int(self._tensor_to_scalar(action))
        steps_until_next_arrival = int(
            self._tensor_to_scalar(getattr(self.env, "steps_until_next_arrival", 0))
        )
        step_count = int(getattr(self.env, "step_count", 0))

        incoming = None
        if getattr(self.env, "car_to_route", None) is not None:
            incoming = self._snap_car(self.env.car_to_route, traveling=False)

        stations: List[StationSnap] = []
        for st in self.env.stations:
            stations.append(
                StationSnap(
                    station_id=int(getattr(st, "id", 0)),
                    nb_chargers=int(getattr(st, "nb_chargers", 0)),
                    max_nb_cars_waiting=int(getattr(st, "max_nb_cars_waiting", 0)),
                    cars_charging=[
                        self._snap_car(c, traveling=False)
                        for c in getattr(st, "cars_charging", [])
                    ],
                    cars_waiting=[
                        self._snap_car(c, traveling=False)
                        for c in getattr(st, "cars_waiting", [])
                    ],
                    cars_traveling=[
                        self._snap_car(c, traveling=True)
                        for c in getattr(st, "cars_traveling", [])
                    ],
                )
            )

        self.snaps.append(
            StepSnap(
                step_count=step_count,
                reward=float(reward),
                action=a,
                steps_until_next_arrival=steps_until_next_arrival,
                incoming=incoming,
                stations=stations,
            )
        )

    # -------------------------
    # Rendering from snapshots
    # -------------------------

    def _render_from_snap(self, snap: StepSnap) -> np.ndarray:
        fig = plt.figure(
            figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi
        )
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.axis("off")

        # Background
        ax.add_patch(
            patches.Rectangle((0, 0), self.width, self.height, color=self.c_bg)
        )

        # Stations
        station_anchors = []
        start_y = self.height - self.margin_y
        for i, st in enumerate(snap.stations):
            cy = start_y - (i * (self.station_h + 20)) - (self.station_h / 2)
            station_anchors.append((self.pos_station_start_x, cy))
            self._draw_station_from_snap(
                ax, st, self.pos_station_start_x, cy, self.pos_station_w, self.station_h
            )

        # Agent
        agent_pos = (self.pos_agent_x, self.height / 2)
        self._draw_agent(ax, agent_pos, snap.action)

        # Roads + traveling cars
        for i, st in enumerate(snap.stations):
            end_pos = (self.pos_station_start_x, station_anchors[i][1])
            is_active_action = (snap.action == i) and (snap.incoming is not None)
            self._draw_road_from_snap(
                ax, agent_pos, end_pos, st.cars_traveling, is_active_action
            )

        # Incoming card / no request
        if snap.incoming is not None:
            self._draw_incoming_card_from_snap(
                ax, self.pos_incoming_x, self.height / 2, snap.incoming
            )
            con = patches.ConnectionPatch(
                (self.pos_incoming_x + 80, self.height / 2),
                (agent_pos[0] - 30, agent_pos[1]),
                "data",
                "data",
                color="white",
                linestyle="--",
                alpha=0.3,
            )
            ax.add_patch(con)
        else:
            self._draw_no_request_from_snap(
                ax, self.pos_incoming_x, self.height / 2, snap.steps_until_next_arrival
            )

        # HUD
        self._draw_hud_from_snap(ax, snap.step_count, snap.reward)

        canvas.draw()
        frame = np.asarray(canvas.buffer_rgba())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return frame

    def _draw_station_from_snap(self, ax, station: StationSnap, x, y, w, h):
        rect = patches.FancyBboxPatch(
            (x, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.01",
            fc=self.c_station_bg,
            ec=self.c_slot_outline,
            zorder=2,
        )
        ax.add_patch(rect)

        header_h = 30
        ax.text(
            x + 15,
            y + h / 2 - 20,
            f"STATION {station.station_id}",
            color="white",
            fontsize=11,
            weight="bold",
            va="center",
        )

        content_y_top = y + h / 2 - header_h
        content_h = h - header_h - 10

        chargers_w = w * 0.65
        self._draw_grid_slots_from_snap(
            ax,
            x + 10,
            content_y_top,
            chargers_w,
            content_h,
            station.nb_chargers,
            station.cars_charging,
            label="CHARGERS",
            icon="⚡",
            is_queue=False,
        )

        queue_x = x + chargers_w + 20
        queue_w = w - chargers_w - 30
        self._draw_grid_slots_from_snap(
            ax,
            queue_x,
            content_y_top,
            queue_w,
            content_h,
            station.max_nb_cars_waiting,
            station.cars_waiting,
            label="QUEUE",
            icon="P",
            is_queue=True,
        )

    def _draw_grid_slots_from_snap(
        self,
        ax,
        x,
        top_y,
        w,
        h,
        total_slots,
        cars: List[CarSnap],
        label,
        icon,
        is_queue,
    ):
        ax.text(x, top_y, label, color=self.c_text_sub, fontsize=8, va="top")

        area_top = top_y - 15
        area_h = h - 15

        if total_slots <= 4:
            rows, cols = total_slots, 1
        elif total_slots <= 8:
            rows, cols = math.ceil(total_slots / 2), 2
        else:
            rows, cols = math.ceil(total_slots / 3), 3

        slot_w = (w - (cols - 1) * 5) / cols
        slot_h = (area_h - (rows - 1) * 5) / rows
        slot_h = min(slot_h, 50)

        for idx in range(total_slots):
            r = idx % rows
            c = idx // rows

            sx = x + c * (slot_w + 5)
            sy = area_top - r * (slot_h + 5) - slot_h

            box = patches.FancyBboxPatch(
                (sx, sy),
                slot_w,
                slot_h,
                boxstyle="round,pad=2",
                fc=self.c_slot_empty,
                ec=self.c_slot_outline,
                alpha=0.5,
            )
            ax.add_patch(box)

            if idx < len(cars):
                car = cars[idx]
                mode = "wait" if is_queue else "charge"
                self._draw_car_mini_from_snap(ax, sx, sy, slot_w, slot_h, car, mode)
            else:
                ax.text(
                    sx + slot_w / 2,
                    sy + slot_h / 2,
                    icon,
                    color=self.c_slot_outline,
                    fontsize=10,
                    ha="center",
                    va="center",
                )

    def _draw_car_mini_from_snap(self, ax, x, y, w, h, car: CarSnap, mode: str):
        if mode == "charge":
            if car.desired_soc > 0 and car.soc >= (car.desired_soc * 0.95):
                color = self.c_car_done
            else:
                color = self.c_car_charge
        else:
            color = self.c_car_wait

        pad = 4
        cw = w - pad * 2
        ch = h - pad * 2
        cx, cy = x + pad, y + pad

        rect = patches.FancyBboxPatch(
            (cx, cy),
            cw,
            ch,
            boxstyle="round,pad=2",
            fc=color,
            ec="white",
            linewidth=1,
            zorder=5,
        )
        ax.add_patch(rect)

        if mode == "charge":
            bx, by = cx + 2, cy + ch / 2 - 2
            bw, bh = cw - 4, 4
            ax.add_patch(
                patches.Rectangle((bx, by), bw, bh, fc="black", alpha=0.3, zorder=6)
            )

            denom = max(car.desired_soc, 0.01)
            fill_w = min(bw, bw * (car.soc / denom))
            ax.add_patch(
                patches.Rectangle((bx, by), fill_w, bh, fc="white", alpha=0.9, zorder=7)
            )

    def _draw_agent(self, ax, pos, action):
        x, y = pos
        ax.add_patch(patches.Circle((x, y), 30, color=self.c_car_neutral, alpha=0.2))
        ax.add_patch(patches.Circle((x, y), 15, color="white", zorder=10))
        ax.text(
            x, y - 35, "AGENT", color="white", ha="center", fontsize=9, weight="bold"
        )

    def _draw_road_from_snap(
        self, ax, start_pos, end_pos, cars_traveling: List[CarSnap], is_active: bool
    ):
        sx, sy = start_pos
        ex, ey = end_pos

        alpha = 0.6 if is_active else 0.2
        width = 3 if is_active else 1.5
        color = "white" if is_active else self.c_road

        ax.plot([sx, ex], [sy, ey], color=color, linewidth=width, alpha=alpha, zorder=1)

        dx = ex - sx
        dy = ey - sy

        for car in cars_traveling:
            t_max = car.max_travel_time
            t_rem = car.travel_time_remaining

            if t_max == 0:
                progress = 1.0
            else:
                progress = 1.0 - (t_rem / t_max)

            cx = sx + dx * progress
            cy = sy + dy * progress

            w, h = 30, 18
            rect = patches.FancyBboxPatch(
                (cx - w / 2, cy - h / 2),
                w,
                h,
                boxstyle="round,pad=1",
                fc=self.c_car_travel,
                ec="white",
                zorder=4,
            )
            ax.add_patch(rect)

            ax.text(
                cx,
                cy,
                f"{int(t_rem)}",
                fontsize=6,
                color="black",
                ha="center",
                va="center",
                weight="bold",
                zorder=5,
            )

    def _draw_incoming_card_from_snap(self, ax, x, y, car: CarSnap):
        w, h = 200, 120
        card = patches.FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=5",
            fc=self.c_station_bg,
            ec=self.c_car_neutral,
            linewidth=2,
            zorder=5,
        )
        ax.add_patch(card)

        ax.text(
            x,
            y + h / 2 - 15,
            "INCOMING REQUEST",
            color="white",
            ha="center",
            fontsize=10,
            weight="bold",
        )

        start_y = y + 10
        self._draw_status_bar(
            ax, x - w / 2 + 10, start_y, w - 20, "SoC", car.soc, car.desired_soc, "cyan"
        )
        self._draw_status_bar(
            ax,
            x - w / 2 + 10,
            start_y - 25,
            w - 20,
            "Urgency",
            car.urgency,
            None,
            "orange",
        )
        self._draw_status_bar(
            ax,
            x - w / 2 + 10,
            start_y - 50,
            w - 20,
            "Capacity",
            car.capacity / self.env.max_car_capacity,
            None,
            "purple",
        )

    def _draw_no_request_from_snap(self, ax, x, y, steps_until_next_arrival: int):
        w, h = 180, 80
        card = patches.FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=5",
            fc=(0.1, 0.1, 0.1),
            ec="#444",
            linestyle="--",
            zorder=1,
        )
        ax.add_patch(card)

        ax.text(
            x,
            y + 10,
            "NO REQUEST",
            color="#666",
            ha="center",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            x,
            y - 10,
            f"Next in: {steps_until_next_arrival} steps",
            color="#888",
            ha="center",
            fontsize=9,
        )

    def _draw_status_bar(self, ax, x, y, w, label, val, target_val, color_name):
        bar_h = 6
        colors = {"cyan": "#3498db", "orange": "#e67e22", "purple": "#9b59b6"}
        c = colors.get(color_name, "white")

        ax.text(x, y, label, color="silver", fontsize=7, va="bottom")

        if label != "Capacity":
            val_txt = f"{int(val * 100)}%"
        else:
            val_txt = f"{int(val * self.env.max_car_capacity)}"

        ax.text(x + w, y, val_txt, color=c, fontsize=7, va="bottom", ha="right")

        # Background track
        ax.add_patch(
            patches.Rectangle((x, y - bar_h - 2), w, bar_h, fc="#333", zorder=6)
        )

        # Fill
        fill_w = w * float(val)
        ax.add_patch(
            patches.Rectangle((x, y - bar_h - 2), fill_w, bar_h, fc=c, zorder=7)
        )

        # Target marker (SoC)
        if target_val is not None:
            tx = x + (w * float(target_val))
            ax.plot([tx, tx], [y - bar_h - 4, y], color="red", linewidth=1.5, zorder=8)

    def _draw_hud_from_snap(self, ax, step_count: int, reward: float):
        ax.text(
            20,
            self.height - 30,
            f"STEP: {step_count}",
            color="white",
            fontsize=16,
            weight="bold",
        )
        ax.text(
            20, self.height - 60, f"REWARD: {reward:.2f}", color="white", fontsize=12
        )

    # -------------------------
    # Save video
    # -------------------------

    def save(self):
        if not self.snaps:
            print("No snapshots to save.")
            return

        print(
            f"Rendering {len(self.snaps)} frames and saving to {self.output_path} ..."
        )

        # Write video directly (no frame list in RAM)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height)
        )

        if not out.isOpened():
            print("Error: Could not open mp4v writer. Falling back to MJPG/AVI.")
            self.output_path = self.output_path.replace(".mp4", ".avi")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            out = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (self.width, self.height)
            )

        if not out.isOpened():
            raise RuntimeError("Could not open video writer (mp4v or MJPG).")

        for snap in self.snaps:
            frame = self._render_from_snap(snap)
            out.write(frame)

        out.release()

        if not os.path.exists(self.output_path):
            print("Error: Video file was not created.")
            return

        print(f"Video saved successfully to {self.output_path}")

        # Optional ffmpeg conversion for browser playback
        if (
            self.convert_with_ffmpeg
            and self.output_path.endswith(".mp4")
            and shutil.which("ffmpeg")
        ):
            print("FFmpeg detected. Converting to web-compatible H.264/yuv420p ...")
            temp_name = self.output_path.replace(".mp4", "_temp.mp4")
            try:
                os.rename(self.output_path, temp_name)
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    temp_name,
                    "-vcodec",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-loglevel",
                    "error",
                    self.output_path,
                ]
                subprocess.run(cmd, check=True)
                os.remove(temp_name)
                print("Conversion complete.")
            except Exception as e:
                print(f"Conversion failed: {e}")
                if os.path.exists(temp_name):
                    os.rename(temp_name, self.output_path)
                print("Kept original video (playable locally).")
        elif self.convert_with_ffmpeg and self.output_path.endswith(".mp4"):
            print(
                "Warning: ffmpeg not found. MP4 may not play in browser. Install with: sudo apt install ffmpeg"
            )

    def reset_recording(self):
        """Drop buffered snapshots and start a fresh segment."""
        self.snaps.clear()

    def save_segment(self, output_path: str):
        """Save current buffered snapshots to a specific path, then keep buffer unless you reset()."""
        old_path = self.output_path
        try:
            self.output_path = output_path
            self.save()
        finally:
            self.output_path = old_path
