# overwrite your gravity-sim/gui.py with this file
"""
gui.py (fixed star layering)

Same features as before. Fixes:
- Background stars are bigger and lowered behind other canvas items.
- Stars are recreated and lowered on reset.
- Lensing + parallax unchanged.
"""

import tkinter as tk
import ttkbootstrap as tb
import numpy as np
from simulation import Body, velocity_verlet_step, handle_collisions, trigger_supernova, G

WIDTH, HEIGHT = 1100, 700
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
MAX_TRAIL_LENGTH = 250
TRAIL_COLOR = "#BFBFBF"


class GravitySimApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gravity Simulator")

        style = tb.Style(theme="darkly")
        self.root.configure(bg=style.colors.bg)

        # Canvas
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Toolbar
        ctrl = tb.Frame(root)
        ctrl.pack(fill="x", pady=6)

        # Buttons
        self.pause_btn = tb.Button(ctrl, text="Pause", bootstyle="outline-pink", command=self.toggle_pause)
        self.pause_btn.pack(side="left", padx=6)

        self.reset_btn = tb.Button(ctrl, text="Reset", bootstyle="outline-secondary", command=self.reset)
        self.reset_btn.pack(side="left", padx=6)

        self.central_var = tk.BooleanVar(value=True)
        self.central_chk = tb.Checkbutton(ctrl, text="Central Mass", variable=self.central_var,
                                          bootstyle="round-toggle", command=self.toggle_central)
        self.central_chk.pack(side="left", padx=6)

        self.add_random_btn = tb.Button(ctrl, text="Add Random Planet", bootstyle="secondary", command=self.add_random_planet)
        self.add_random_btn.pack(side="left", padx=6)

        self.add_binary_btn = tb.Button(ctrl, text="Add Binary System", bootstyle="warning", command=self.add_binary_system)
        self.add_binary_btn.pack(side="left", padx=6)

        self.add_bh_btn = tb.Button(ctrl, text="Add Black Hole", bootstyle="danger-outline", command=self.add_blackhole_button)
        self.add_bh_btn.pack(side="left", padx=6)

        self.supernova_btn = tb.Button(ctrl, text="Trigger Supernova", bootstyle="primary", command=self.trigger_supernova_button)
        self.supernova_btn.pack(side="left", padx=6)

        self.add_custom_btn = tb.Button(ctrl, text="Add Custom Planet", bootstyle="info", command=self.open_custom_dialog)
        self.add_custom_btn.pack(side="left", padx=6)

        self.clear_trails_btn = tb.Button(ctrl, text="Clear Trails", bootstyle="secondary", command=self.clear_trails)
        self.clear_trails_btn.pack(side="left", padx=6)

        # speed slider on right
        self.speed_var = tk.DoubleVar(value=1.0)
        tb.Label(ctrl, text="Speed", bootstyle="inverse-dark").pack(side="right", padx=8)
        tb.Scale(ctrl, from_=0.1, to=5.0, variable=self.speed_var, orient="horizontal", length=200, bootstyle="info").pack(side="right", padx=6)

        # sim state
        self.bodies = []
        self.dt = 1e-3
        self.running = True

        # camera
        self.scale = 100.0
        self.offset_x = 0.0
        self.offset_y = 0.0

        # background stars as (id, world_x, world_y, parallax)
        self.bg_stars = []

        # trails / flash / supernova
        self.trails = {}
        self.flash_timers = {}
        self.supernova_flash = {}

        # bind events
        self._bind_events()

        # create and lower background stars BEFORE reset so they are behind new objects
        self._create_background_stars(350)
        self._lower_background_tag()

        # initial scene
        self.reset(initial_setup=True)

        # start loop
        self.root.after(10, self.update_loop)

    # ------------- background stars / parallax / lensing -------------
    def _create_background_stars(self, n=300):
        # remove existing stars
        for dot, *_ in list(self.bg_stars):
            try:
                self.canvas.delete(dot)
            except Exception:
                pass
        self.bg_stars.clear()

        # world region slightly larger than viewport
        world_w = WIDTH / self.scale * 2.5
        world_h = HEIGHT / self.scale * 2.5
        for _ in range(n):
            wx = np.random.uniform(-world_w / 2, world_w / 2)
            wy = np.random.uniform(-world_h / 2, world_h / 2)
            p = float(np.random.uniform(0.15, 0.85))
            sx, sy = self.world_to_screen(wx, wy, parallax=p)
            # make stars slightly larger (2-3 px) so they're visible on HiDPI / different displays
            size = np.random.choice([1.5, 2.0, 2.5])
            dot = self.canvas.create_oval(sx, sy, sx + size, sy + size, fill="white", outline="")
            # tag for lowering
            self.canvas.addtag_withtag("bg", dot)
            self.bg_stars.append((dot, wx, wy, p))

        if not self.bg_stars:
            print("Warning: no background stars were created (n=0).")
        # ensure they are beneath everything
        self._lower_background_tag()

    def _lower_background_tag(self):
        try:
            self.canvas.tag_lower("bg")
        except Exception:
            pass

    def world_to_screen(self, wx, wy, parallax=1.0):
        """Convert world coordinates to screen coordinates applying parallax."""
        sx = wx * self.scale + CENTER_X + self.offset_x * parallax
        sy = wy * self.scale + CENTER_Y + self.offset_y * parallax
        return sx, sy

    def apply_gravitational_lensing(self):
        # update star positions (screen-space) and apply small lensing near blackholes
        for dot_id, wx, wy, p in list(self.bg_stars):
            sx, sy = self.world_to_screen(wx, wy, parallax=p)
            rel_x, rel_y = sx, sy
            for b in self.bodies:
                if b.body_type == "blackhole":
                    bx, by = self.world_to_screen(b.position[0], b.position[1], parallax=1.0)
                    dx = rel_x - bx
                    dy = rel_y - by
                    dist2 = dx * dx + dy * dy
                    if dist2 < (450 ** 2):
                        strength = (b.mass + 1.0) / (dist2 + 1.0)
                        rel_x += dx * 0.0009 * strength
                        rel_y += dy * 0.0009 * strength
            try:
                # update coords of the dot; keep small size
                size = 2
                self.canvas.coords(dot_id, rel_x, rel_y, rel_x + size, rel_y + size)
            except Exception:
                pass

    # ------------- events (pan/zoom) -------------
    def _bind_events(self):
        self.canvas.bind("<Button-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_move)
        self.canvas.bind("<MouseWheel>", self._on_zoom_windows)
        self.canvas.bind("<Button-4>", self._on_zoom_linux)
        self.canvas.bind("<Button-5>", self._on_zoom_linux)

    def _on_pan_start(self, event):
        self._pan_start = (event.x, event.y)
        self._pan_offset_start = (self.offset_x, self.offset_y)

    def _on_pan_move(self, event):
        if not hasattr(self, "_pan_start") or self._pan_start is None:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self.offset_x = self._pan_offset_start[0] + dx
        self.offset_y = self._pan_offset_start[1] + dy

    def _on_zoom_windows(self, event):
        old_scale = self.scale
        factor = 1.1 if event.delta > 0 else 0.9
        wx = (event.x - CENTER_X - self.offset_x) / old_scale
        wy = (event.y - CENTER_Y - self.offset_y) / old_scale
        self.scale *= factor
        sx = wx * self.scale + CENTER_X + self.offset_x
        sy = wy * self.scale + CENTER_Y + self.offset_y
        self.offset_x += event.x - sx
        self.offset_y += event.y - sy

    def _on_zoom_linux(self, event):
        old_scale = self.scale
        factor = 1.1 if event.num == 4 else 0.9
        wx = (event.x - CENTER_X - self.offset_x) / old_scale
        wy = (event.y - CENTER_Y - self.offset_y) / old_scale
        self.scale *= factor
        sx = wx * self.scale + CENTER_X + self.offset_x
        sy = wy * self.scale + CENTER_Y + self.offset_y
        self.offset_x += event.x - sx
        self.offset_y += event.y - sy

    # ------------- body management (unchanged) -------------
    def add_body(self, pos, vel, mass, radius, color, body_type="planet"):
        b = Body(pos, vel, mass, radius=radius, color=color, body_type=body_type)
        b.id = self.canvas.create_oval(0, 0, 1, 1, fill=color, outline="")
        b.trail = []
        b._last_screen = None
        self.bodies.append(b)
        return b

    def add_random_planet(self):
        central = next((x for x in self.bodies if x.body_type == "star" and x.color == "yellow"), None)
        if central is None:
            pos = [np.random.uniform(-3, 3), np.random.uniform(-3, 3)]
            vel = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
            self.add_body(pos, vel, mass=1.0, radius=0.05, color="cyan")
            return

        r = np.random.uniform(1.5, 4.0)
        theta = np.random.uniform(0, 2 * np.pi)
        x = central.position[0] + r * np.cos(theta)
        y = central.position[1] + r * np.sin(theta)
        v_circ = np.sqrt(G * central.mass / r)
        vx = -v_circ * np.sin(theta)
        vy = v_circ * np.cos(theta)
        color = np.random.choice(["cyan", "pink", "lightblue", "orange", "white", "green"])
        self.add_body([x, y], [vx, vy], mass=1.0, radius=0.05, color=color)

    def add_binary_system(self):
        m = 8.0
        sep = 1.6
        total = m * 2.0
        omega = np.sqrt(G * total / (sep ** 3))
        v1 = omega * (sep * 0.5)
        v2 = v1
        self.add_body([-sep * 0.5, 0.0], [0.0, v1], mass=m, radius=0.07, color="pink", body_type="star")
        self.add_body([sep * 0.5, 0.0], [0.0, -v2], mass=m, radius=0.07, color="lightblue", body_type="star")

    def add_blackhole_button(self):
        self.add_body([0.0, 0.0], [0.0, 0.0], mass=200.0, radius=0.08, color="black", body_type="blackhole")

    def trigger_supernova_button(self):
        stars = [b for b in self.bodies if b.body_type == "star"]
        if not stars:
            return
        target = max(stars, key=lambda s: s.mass)
        self.supernova_flash[target] = 18

    def open_custom_dialog(self):
        dlg = tb.Toplevel(self.root)
        dlg.title("Add Custom Planet")
        frm = tb.Frame(dlg, padding=8)
        frm.pack(fill="both", expand=True)
        labels = [("x", "0.0"), ("y", "2.0"), ("vx", "0.0"), ("vy", "0.0"), ("mass", "1.0"), ("radius", "0.05")]
        inputs = {}
        for i, (lab, val) in enumerate(labels):
            tb.Label(frm, text=lab).grid(row=i, column=0, sticky="w", padx=4, pady=2)
            e = tb.Entry(frm); e.grid(row=i, column=1, padx=4, pady=2); e.insert(0, val)
            inputs[lab] = e
        tb.Label(frm, text="color").grid(row=len(labels), column=0, sticky="w", padx=4, pady=2)
        color_var = tk.StringVar(value="cyan")
        colors = ["cyan", "pink", "lightblue", "orange", "white", "green", "yellow", "magenta", "black"]
        comb = tb.Combobox(frm, textvariable=color_var, values=colors, state="readonly")
        comb.grid(row=len(labels), column=1, padx=4, pady=2)
        def on_add():
            try:
                x = float(inputs["x"].get()); y = float(inputs["y"].get())
                vx = float(inputs["vx"].get()); vy = float(inputs["vy"].get())
                mass = float(inputs["mass"].get()); radius = float(inputs["radius"].get())
                color = color_var.get()
            except Exception:
                tb.Label(frm, text="Invalid input", bootstyle="danger").grid(row=len(labels)+1, column=0, columnspan=2)
                return
            self.add_body([x, y], [vx, vy], mass=mass, radius=radius, color=color)
            dlg.destroy()
        tb.Button(frm, text="Add", bootstyle="success", command=on_add).grid(row=len(labels)+1, column=0, pady=6)
        tb.Button(frm, text="Cancel", bootstyle="secondary", command=dlg.destroy).grid(row=len(labels)+1, column=1, pady=6)

    def clear_trails(self):
        for b in list(self.bodies):
            for tid in getattr(b, "trail", []):
                try:
                    self.canvas.delete(tid)
                except Exception:
                    pass
            b.trail = []
            b._last_screen = None

    # ------------- main loop -------------
    def update_loop(self):
        try:
            if self.running and self.bodies:
                self.step()
            self.draw()
        except Exception as e:
            print("Simulation error:", e)
        self.root.after(16, self.update_loop)

    def step(self):
        dt_eff = self.dt * self.speed_var.get()
        velocity_verlet_step(self.bodies, dt_eff)
        new_bodies, flashes = handle_collisions(self.bodies)
        removed = [b for b in self.bodies if b not in new_bodies]
        for b in removed:
            if hasattr(b, "id") and b.id:
                try:
                    self.canvas.delete(b.id)
                except Exception:
                    pass
            for tid in getattr(b, "trail", []):
                try:
                    self.canvas.delete(tid)
                except Exception:
                    pass
        for b in new_bodies:
            if not hasattr(b, "id") or b.id is None:
                b.id = self.canvas.create_oval(0, 0, 1, 1, fill=b.color, outline="")
                b.trail = []
                b._last_screen = None
        self.bodies = new_bodies
        for merged_body, final_color in flashes:
            if hasattr(merged_body, "id") and merged_body.id:
                self.canvas.itemconfig(merged_body.id, fill="white")
                self.flash_timers[merged_body.id] = {"ticks": 18, "final_color": final_color}
        # supernova handling
        exploded = []
        for body, ticks in list(self.supernova_flash.items()):
            if ticks > 0:
                if hasattr(body, "id") and body.id:
                    try:
                        self.canvas.itemconfig(body.id, fill="white")
                    except Exception:
                        pass
                self.supernova_flash[body] -= 1
            else:
                frags = trigger_supernova(self.bodies, body)
                for f in frags:
                    f.id = self.canvas.create_oval(0, 0, 1, 1, fill=f.color, outline="")
                    f.trail = []
                    f._last_screen = None
                    self.bodies.append(f)
                exploded.append(body)
        for b in exploded:
            if b in self.supernova_flash:
                del self.supernova_flash[b]

    def draw(self):
        # draw bodies + trails (trails in neutral gray)
        for b in list(self.bodies):
            sx, sy = self.world_to_screen(b.position[0], b.position[1], parallax=1.0)
            rpx = max(1, int(round(b.radius * self.scale)))
            try:
                self.canvas.coords(b.id, sx - rpx, sy - rpx, sx + rpx, sy + rpx)
            except Exception:
                pass
            if b.id in self.flash_timers:
                st = self.flash_timers[b.id]
                st["ticks"] -= 1
                try:
                    self.canvas.itemconfig(b.id, fill="white")
                except Exception:
                    pass
                if st["ticks"] <= 0:
                    try:
                        self.canvas.itemconfig(b.id, fill=st["final_color"])
                    except Exception:
                        pass
                    del self.flash_timers[b.id]
            if getattr(b, "trail", None) is None:
                b.trail = []
            last = getattr(b, "_last_screen", None)
            if last is not None:
                lx, ly = last
            else:
                lx, ly = sx, sy
            seg = self.canvas.create_line(lx, ly, sx, sy, fill=TRAIL_COLOR, width=1)
            b.trail.append(seg)
            b._last_screen = (sx, sy)
            if len(b.trail) > MAX_TRAIL_LENGTH:
                try:
                    self.canvas.delete(b.trail.pop(0))
                except Exception:
                    pass
        # lens background stars
        self.apply_gravitational_lensing()
        # ensure bg stars remain at bottom (in case new canvas items were created above them)
        self._lower_background_tag()

    def toggle_pause(self):
        self.running = not self.running
        self.pause_btn.config(text="Resume" if not self.running else "Pause")

    def reset(self, initial_setup=False):
        # delete all canvas items (bodies & trails)
        for b in list(self.bodies):
            if hasattr(b, "id") and b.id:
                try:
                    self.canvas.delete(b.id)
                except Exception:
                    pass
            for tid in getattr(b, "trail", []):
                try:
                    self.canvas.delete(tid)
                except Exception:
                    pass
        # delete and recreate background stars
        for dot, _, _, _ in list(self.bg_stars):
            try:
                self.canvas.delete(dot)
            except Exception:
                pass
        self.bg_stars = []
        self._create_background_stars(350)
        self._lower_background_tag()
        # reset sim state
        self.bodies = []
        self.trails = {}
        self.flash_timers = {}
        self.supernova_flash = {}
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.scale = 100.0
        # add central + initial orbiters
        if self.central_var.get():
            self.add_body([0.0, 0.0], [0.0, 0.0], mass=50.0, radius=0.2, color="yellow", body_type="star")
        if initial_setup:
            central = next((b for b in self.bodies if b.body_type == "star"), None)
            if central:
                radii = [1.0, 1.6, 2.4, 3.2]
                colors = ["cyan", "pink", "lightblue", "orange"]
                for rr, col in zip(radii, colors):
                    theta = np.random.uniform(0, 2 * np.pi)
                    x = central.position[0] + rr * np.cos(theta)
                    y = central.position[1] + rr * np.sin(theta)
                    v = np.sqrt(G * central.mass / rr)
                    vx = -v * np.sin(theta)
                    vy = v * np.cos(theta)
                    self.add_body([x, y], [vx, vy], mass=1.0, radius=0.05, color=col)

    def toggle_central(self):
        if not self.central_var.get():
            removed = [b for b in list(self.bodies) if getattr(b, "color", "") == "yellow" and b.body_type == "star"]
            for b in removed:
                try:
                    if hasattr(b, "id"): self.canvas.delete(b.id)
                except Exception:
                    pass
                for tid in getattr(b, "trail", []):
                    try:
                        self.canvas.delete(tid)
                    except Exception:
                        pass
                if b in self.bodies:
                    self.bodies.remove(b)
        else:
            self.add_body([0.0, 0.0], [0.0, 0.0], mass=50.0, radius=0.2, color="yellow", body_type="star")


if __name__ == "__main__":
    root = tb.Window(themename="darkly")
    app = GravitySimApp(root)
    root.mainloop()
