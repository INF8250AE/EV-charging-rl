import cv2
import numpy as np
import torch
import os
import shutil
import subprocess
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg

class EvVizRecorder:
    def __init__(self, env, output_path="videos/ev_rollout.mp4", fps=2):
        self.env = env
        self.output_path = output_path
        self.fps = fps
        self.frames = []
        
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
        self.c_car_neutral = (0.7, 0.7, 0.75)    # Gray (Routing)
        self.c_car_travel = (0.95, 0.75, 0.2)    # Yellow (Traveling)
        self.c_car_wait = (0.85, 0.3, 0.3)       # Red (Waiting)
        self.c_car_charge = (0.2, 0.6, 0.9)      # Blue (Charging)
        self.c_car_done = (0.3, 0.8, 0.4)        # Green (Done)

        self.nb_stations = len(env.stations)
        # Dynamic height calculation with padding
        self.margin_y = 60
        self.available_h = self.height - (self.margin_y * 2)
        self.station_h = min(180, self.available_h / self.nb_stations - 20) 
        
        self.pos_incoming_x = self.width * 0.10
        self.pos_agent_x = self.width * 0.30
        self.pos_station_start_x = self.width * 0.60
        self.pos_station_w = self.width * 0.35
        
    def _tensor_to_scalar(self, val):
        if isinstance(val, torch.Tensor):
            return val.item()
        return val

    def record_step(self, prev_obs, obs, action, reward, done):
        """Render the complete frame."""
        
        fig = plt.figure(figsize=(self.width/self.dpi, self.height/self.dpi), dpi=self.dpi)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.axis('off')
        
        # Background
        ax.add_patch(patches.Rectangle((0, 0), self.width, self.height, color=self.c_bg))
        
        # 1. Draw Stations (Right Side)
        station_anchors = []
        start_y = self.height - self.margin_y
        
        for i, station in enumerate(self.env.stations):
            # Top-down positioning
            cy = start_y - (i * (self.station_h + 20)) - (self.station_h / 2)
            station_anchors.append((self.pos_station_start_x, cy))
            
            self._draw_station(ax, station, self.pos_station_start_x, cy, self.pos_station_w, self.station_h)

        # 2. Draw Agent (Center Node)
        agent_pos = (self.pos_agent_x, self.height / 2)
        self._draw_agent(ax, agent_pos, action)

        # 3. Draw Roads (Agent -> Stations)
        for i, station in enumerate(self.env.stations):
            end_pos = (self.pos_station_start_x, station_anchors[i][1])
            # Check if this road was just chosen by the agent
            is_active_action = (self._tensor_to_scalar(action) == i) and (self.env.car_to_route is not None)
            
            self._draw_road(ax, agent_pos, end_pos, station.cars_traveling, is_active_action)

        # 4. Draw Incoming Request
        if self.env.car_to_route is not None:
            self._draw_incoming_card(ax, self.pos_incoming_x, self.height/2, self.env.car_to_route)
            
            # Connection line to agent
            con = patches.ConnectionPatch(
                (self.pos_incoming_x + 80, self.height/2), 
                (agent_pos[0] - 30, agent_pos[1]), 
                "data", "data", color="white", linestyle="--", alpha=0.3
            )
            ax.add_patch(con)
        else:
            self._draw_no_request(ax, self.pos_incoming_x, self.height/2)

        # 5. HUD / Info
        self._draw_hud(ax, reward)

        canvas.draw()
        buf = canvas.buffer_rgba()
        frame = np.asarray(buf)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        self.frames.append(frame)
        plt.close(fig)

    def _draw_station(self, ax, station, x, y, w, h):
        rect = patches.FancyBboxPatch((x, y - h/2), w, h, boxstyle="round,pad=0.01", 
                                      fc=self.c_station_bg, ec=self.c_slot_outline, zorder=2)
        ax.add_patch(rect)
        
        header_h = 30
        ax.text(x + 15, y + h/2 - 20, f"STATION {station.id}", 
                color="white", fontsize=11, weight="bold", va="center")
        
        content_y_top = y + h/2 - header_h
        content_h = h - header_h - 10
        
        chargers_w = w * 0.65
        self._draw_grid_slots(ax, x + 10, content_y_top, chargers_w, content_h, 
                              station.nb_chargers, station.cars_charging, 
                              label="CHARGERS", icon="⚡", is_queue=False)

        queue_x = x + chargers_w + 20
        queue_w = w - chargers_w - 30
        self._draw_grid_slots(ax, queue_x, content_y_top, queue_w, content_h, 
                              station.max_nb_cars_waiting, station.cars_waiting, 
                              label="QUEUE", icon="P", is_queue=True)

    def _draw_grid_slots(self, ax, x, top_y, w, h, total_slots, cars, label, icon, is_queue):
        ax.text(x, top_y, label, color=self.c_text_sub, fontsize=8, va="top")
        
        area_top = top_y - 15
        area_h = h - 15
        
        if total_slots <= 4:
            rows, cols = total_slots, 1
        elif total_slots <= 8:
            rows, cols = math.ceil(total_slots/2), 2
        else:
            rows, cols = math.ceil(total_slots/3), 3
            
        slot_w = (w - (cols-1)*5) / cols
        slot_h = (area_h - (rows-1)*5) / rows

        slot_h = min(slot_h, 50) 
        
        for idx in range(total_slots):
            r = idx % rows
            c = idx // rows
            
            sx = x + c * (slot_w + 5)
            sy = area_top - r * (slot_h + 5) - slot_h
            
            box = patches.FancyBboxPatch((sx, sy), slot_w, slot_h, 
                                         boxstyle="round,pad=2", fc=self.c_slot_empty, 
                                         ec=self.c_slot_outline, alpha=0.5)
            ax.add_patch(box)
            
            if idx < len(cars):
                car = cars[idx]
                mode = "wait" if is_queue else "charge"
                self._draw_car_mini(ax, sx, sy, slot_w, slot_h, car, mode)
            else:
                ax.text(sx + slot_w/2, sy + slot_h/2, icon, 
                        color=self.c_slot_outline, fontsize=10, ha="center", va="center")

    def _draw_car_mini(self, ax, x, y, w, h, car, mode):
        """Draws a simplified car inside a station slot."""
        color = self.c_car_wait
        if mode == "charge":
            soc = self._tensor_to_scalar(car.soc)
            des = self._tensor_to_scalar(car.desired_soc)
            # Green if >= 95% of target
            if des > 0 and soc >= (des * 0.95):
                color = self.c_car_done
            else:
                color = self.c_car_charge
        pad = 4
        cw = w - pad*2
        ch = h - pad*2
        cx, cy = x + pad, y + pad
        
        rect = patches.FancyBboxPatch((cx, cy), cw, ch, boxstyle="round,pad=2", 
                                      fc=color, ec="white", linewidth=1, zorder=5)
        ax.add_patch(rect)
        
        if mode == "charge":
            soc = self._tensor_to_scalar(car.soc)
            des = self._tensor_to_scalar(car.desired_soc)
            
            bx, by = cx + 2, cy + ch/2 - 2
            bw, bh = cw - 4, 4
            ax.add_patch(patches.Rectangle((bx, by), bw, bh, fc="black", alpha=0.3, zorder=6))

            fill_w = bw * (soc / max(des, 0.01))
            fill_w = min(fill_w, bw)
            ax.add_patch(patches.Rectangle((bx, by), fill_w, bh, fc="white", alpha=0.9, zorder=7))

    def _draw_agent(self, ax, pos, action):
        x, y = pos

        glow = patches.Circle((x, y), 30, color=self.c_car_neutral, alpha=0.2)
        ax.add_patch(glow)

        circle = patches.Circle((x, y), 15, color="white", zorder=10)
        ax.add_patch(circle)
        ax.text(x, y - 35, "AGENT", color="white", ha="center", fontsize=9, weight="bold")

    def _draw_road(self, ax, start_pos, end_pos, cars_traveling, is_active):
        sx, sy = start_pos
        ex, ey = end_pos
        
        alpha = 0.6 if is_active else 0.2
        width = 3 if is_active else 1.5
        color = "white" if is_active else self.c_road
        
        ax.plot([sx, ex], [sy, ey], color=color, linewidth=width, alpha=alpha, zorder=1)
        
        # Draw Traveling Cars
        dx = ex - sx
        dy = ey - sy
        
        for car in cars_traveling:
            t_max = self._tensor_to_scalar(car.max_travel_time)
            t_rem = self._tensor_to_scalar(car.travel_time_remaining)
            
            if t_max == 0: progress = 1.0
            else: progress = 1.0 - (t_rem / t_max)
            
            cx = sx + dx * progress
            cy = sy + dy * progress
            
            w, h = 30, 18
            rect = patches.FancyBboxPatch((cx - w/2, cy - h/2), w, h, boxstyle="round,pad=1", 
                                          fc=self.c_car_travel, ec="white", zorder=4)
            ax.add_patch(rect)
            
            ax.text(cx, cy, f"{int(t_rem)}", fontsize=6, color="black", ha="center", va="center", weight="bold", zorder=5)

    def _draw_incoming_card(self, ax, x, y, car):
        w, h = 200, 120
        card = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=5", 
                                      fc=self.c_station_bg, ec=self.c_car_neutral, linewidth=2, zorder=5)
        ax.add_patch(card)
        
        ax.text(x, y + h/2 - 15, "INCOMING REQUEST", color="white", ha="center", fontsize=10, weight="bold")
        
        soc = self._tensor_to_scalar(car.soc)
        des = self._tensor_to_scalar(car.desired_soc)
        cap = self._tensor_to_scalar(car.capacity)
        urg = self._tensor_to_scalar(car.urgency)
        
        start_y = y + 10
        self._draw_status_bar(ax, x - w/2 + 10, start_y, w - 20, "SoC", soc, des, "cyan")
        self._draw_status_bar(ax, x - w/2 + 10, start_y - 25, w - 20, "Urgency", urg, None, "orange")
        self._draw_status_bar(ax, x - w/2 + 10, start_y - 50, w - 20, "Capacity", cap/self.env.max_car_capacity, None, "purple")

    def _draw_no_request(self, ax, x, y):
        """Placeholder when no car is waiting."""
        w, h = 180, 80
        card = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=5", 
                                      fc=(0.1, 0.1, 0.1), ec="#444", linestyle="--", zorder=1)
        ax.add_patch(card)
        
        steps = self._tensor_to_scalar(self.env.steps_until_next_arrival)
        ax.text(x, y + 10, "NO REQUEST", color="#666", ha="center", fontsize=10, weight="bold")
        ax.text(x, y - 10, f"Next in: {steps} steps", color="#888", ha="center", fontsize=9)

    def _draw_status_bar(self, ax, x, y, w, label, val, target_val, color_name):
        bar_h = 6
        colors = {"cyan": "#3498db", "orange": "#e67e22", "purple": "#9b59b6"}
        c = colors.get(color_name, "white")
        

        ax.text(x, y, label, color="silver", fontsize=7, va="bottom")
 
        val_txt = f"{int(val*100)}%" if label != "Capacity" else f"{int(val * self.env.max_car_capacity)}"
        ax.text(x + w, y, val_txt, color=c, fontsize=7, va="bottom", ha="right")
        
        # Background Track
        ax.add_patch(patches.Rectangle((x, y - bar_h - 2), w, bar_h, fc="#333", zorder=6))
        
        fill_w = w * val
        ax.add_patch(patches.Rectangle((x, y - bar_h - 2), fill_w, bar_h, fc=c, zorder=7))
        
        # Target Marker (for SoC)
        if target_val is not None:
            tx = x + (w * target_val)
            ax.plot([tx, tx], [y - bar_h - 4, y], color="red", linewidth=1.5, zorder=8)

    def _draw_hud(self, ax, reward):
        """Top left info."""
        ax.text(20, self.height - 30, f"STEP: {self.env.step_count}", color="white", fontsize=16, weight="bold")
        ax.text(20, self.height - 60, f"REWARD: {reward:.2f}", color="white", fontsize=12)

    def save(self):
        if not self.frames:
            print("No frames to save.")
            return

        print(f"Saving video to {self.output_path}...")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))

        if not out.isOpened():
            print("Error: Could not open video writer. Trying default backend...")
            self.output_path = self.output_path.replace(".mp4", ".avi")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))

        for frame in self.frames:
            out.write(frame)

        out.release()
        
        if not os.path.exists(self.output_path):
            print("Error: Video file was not created.")
            return

        print(f"Video saved successfully to {self.output_path} (Codec: mp4v)")

        # We check if ffmpeg is available in the system environment
        if shutil.which("ffmpeg"):
            print("FFmpeg detected. Converting to web-compatible format...")
            
            temp_name = self.output_path.replace(".mp4", "_temp.mp4")
            
            try:
                # Rename original to temp
                os.rename(self.output_path, temp_name)
                
                # Run ffmpeg via Python subprocess (cleaner than os.system)
                # -y: overwrite
                # -vcodec libx264: Force H.264
                # -pix_fmt yuv420p: Force pixel format for web
                cmd = [
                    "ffmpeg", "-y", 
                    "-i", temp_name,
                    "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-loglevel", "error", # Quiet mode
                    self.output_path
                ]
                
                subprocess.run(cmd, check=True)
                
                os.remove(temp_name)
                print("Conversion complete: Video is now Web compatible.")
                
            except Exception as e:
                print(f"Conversion failed: {e}")
                if os.path.exists(temp_name):
                    os.rename(temp_name, self.output_path)
                print("Kept original video (Playable in VLC/Local Player).")
        else:
            print("Warning: FFmpeg not found. Video saved but may not play in Browser.")
            print("To fix compatibility: 'sudo apt install ffmpeg' or use the 'moviepy' library.")