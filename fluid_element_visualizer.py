# fluid_element_visualizer.py
import streamlit as st
import sympy as sp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import streamlit.components.v1 as components

# Make matplotlib use jshtml for animations (avoid ffmpeg)
mpl.rcParams['animation.html'] = 'jshtml'

st.set_page_config(page_title="Rectangular Fluid Element Visualizer", layout="wide")
st.title("Rectangular Fluid Element Visualizer (Option A — Manual formulas)")

st.markdown(
    "Enter 2D velocity components `u(x,y)` and `v(x,y)` (SymPy-style). "
    "App checks steady / constant-density assumptions, computes Bernoulli pressure difference (if valid), "
    "and animates a rectangular fluid element deforming in the flow."
)

# -------------------------
# User inputs (left column)
# -------------------------
left, right = st.columns([1, 1])

with left:
    st.header("Velocity field (enter expressions)")
    u_str = st.text_input("u(x, y) =", value="1 - 0.5*y")
    v_str = st.text_input("v(x, y) =", value="0.5*x")

    st.header("Density (ρ)")
    rho_str = st.text_input("ρ (constant):", value="1.0")

    st.header("Pressure difference points")
    p1x = st.number_input("p1: x₁", value=0.0, format="%.6f")
    p1y = st.number_input("p1: y₁", value=0.0, format="%.6f")
    p2x = st.number_input("p2: x₂", value=1.0, format="%.6f")
    p2y = st.number_input("p2: y₂", value=0.5, format="%.6f")

with right:
    st.header("Rectangular fluid element")
    rect_x0 = st.number_input("Rectangle start x", value=0.2, format="%.6f")
    rect_y0 = st.number_input("Rectangle start y", value=0.0, format="%.6f")
    rect_w = st.number_input("Width", value=0.4, format="%.6f")
    rect_h = st.number_input("Height", value=0.2, format="%.6f")
    grid_nx = st.number_input("Grid points (x-direction)", min_value=2, value=6, step=1)
    grid_ny = st.number_input("Grid points (y-direction)", min_value=2, value=4, step=1)

    st.header("Simulation settings")
    t_final = st.number_input("Simulation time (s)", min_value=0.1, value=4.0, format="%.4f")
    dt = st.number_input("Time step (s)", min_value=1e-4, value=0.05, format="%.4f")

st.markdown("---")

# -------------------------
# Parse symbolic expressions
# -------------------------
x, y, t = sp.symbols("x y t")
parse_error = None
try:
    u_expr = sp.sympify(u_str)
    v_expr = sp.sympify(v_str)
    rho_expr = sp.sympify(rho_str)
except Exception as e:
    parse_error = str(e)

if parse_error:
    st.error("Error parsing expressions: " + parse_error)
    st.stop()

# Lambdify to numeric functions
# Use 'numpy' to allow vectorized evaluation later if needed
u_func = sp.lambdify((x, y), u_expr, "numpy")
v_func = sp.lambdify((x, y), v_expr, "numpy")
rho_func = sp.lambdify((x, y), rho_expr, "numpy")

# -------------------------
# Assumption checks
# -------------------------
st.subheader("Assumption checks")
steady = ("t" not in u_str) and ("t" not in v_str)
const_density = ("x" not in rho_str) and ("y" not in rho_str) and ("t" not in rho_str)

st.write("- Steady flow (no time dependence detected):", "Yes" if steady else "No")
st.write("- Constant density (ρ independent of x,y,t):", "Yes" if const_density else "No")
st.write("- Inviscid: assumed")

if not steady:
    st.warning("Velocity expressions contain 't' or appear time-dependent. Bernoulli may not apply.")
if not const_density:
    st.warning("Density appears variable. Bernoulli may not apply.")

# -------------------------
# Bernoulli pressure diff
# -------------------------
if steady and const_density:
    try:
        rho_val = float(rho_func(0.0, 0.0))
    except Exception:
        # fallback: evaluate at p1
        try:
            rho_val = float(rho_func(p1x, p1y))
        except Exception:
            rho_val = None

    try:
        u1 = float(u_func(p1x, p1y))
        v1 = float(v_func(p1x, p1y))
        u2 = float(u_func(p2x, p2y))
        v2 = float(v_func(p2x, p2y))
        mag1_sq = u1**2 + v1**2
        mag2_sq = u2**2 + v2**2
        if rho_val is not None:
            dp_val = 0.5 * rho_val * (mag2_sq - mag1_sq)
            st.subheader("Pressure difference (Bernoulli, p1 - p2)")
            st.success(f"p₁ - p₂ = {dp_val:.6g}  (using ρ = {rho_val})")
            st.write(f"Velocity at p1 = ({u1:.6g}, {v1:.6g}), |u1|^2 = {mag1_sq:.6g}")
            st.write(f"Velocity at p2 = ({u2:.6g}, {v2:.6g}), |u2|^2 = {mag2_sq:.6g}")
        else:
            st.error("Could not evaluate density numerically; pressure difference not computed.")
    except Exception as e:
        st.error("Error computing pressure difference: " + str(e))
else:
    st.info("Bernoulli pressure difference only computed when flow is steady and density is constant.")

st.markdown("---")

# -------------------------
# Prepare rectangular element
# -------------------------
nx = int(grid_nx)
ny = int(grid_ny)

xs = np.linspace(rect_x0 - rect_w / 2.0, rect_x0 + rect_w / 2.0, nx)
ys = np.linspace(rect_y0 - rect_h / 2.0, rect_y0 + rect_h / 2.0, ny)
X0, Y0 = np.meshgrid(xs, ys)
points0 = np.vstack([X0.ravel(), Y0.ravel()]).T
npoints = points0.shape[0]

# Time vector
time_vec = np.arange(0.0, t_final + 1e-12, dt)

# --------------------------------
# Particle advection helper
# --------------------------------
def advect_particle(p0):
    """Integrate a single particle path (returns array shape (nt,2))."""
    def rhs(pos, tau):
        # pos = [x, y]
        return [float(u_func(pos[0], pos[1])), float(v_func(pos[0], pos[1]))]
    sol = odeint(rhs, p0, time_vec)
    return sol

# Integrate all particles (can be slow for large grids — give feedback)
with st.spinner("Integrating particle paths..."):
    positions = np.zeros((len(time_vec), npoints, 2))
    for i, p in enumerate(points0):
        try:
            traj = advect_particle(p)
        except Exception as e:
            st.error(f"Error integrating particle {i}: {e}")
            st.stop()
        positions[:, i, :] = traj

# -------------------------
# Create animation figure
# -------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal', 'box')

# Set axis limits with margins
all_x = positions[:, :, 0]
all_y = positions[:, :, 1]
xmin, xmax = all_x.min(), all_x.max()
ymin, ymax = all_y.min(), all_y.max()
xpad = max(0.1, 0.1 * (xmax - xmin + 1e-12))
ypad = max(0.1, 0.1 * (ymax - ymin + 1e-12))
ax.set_xlim(xmin - xpad, xmax + xpad)
ax.set_ylim(ymin - ypad, ymax + ypad)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Rectangular Fluid Element — Lagrangian Advection")

# Plot initial scatter and material lines
scatter = ax.scatter([], [], s=20)
rows = []
cols = []
for j in range(ny):
    row_idx = list(range(j * nx, j * nx + nx))
    rows.append(row_idx)
for i in range(nx):
    col_idx = list(range(i, nx * ny, nx))
    cols.append(col_idx)

row_lines = [ax.plot([], [], lw=1)[0] for _ in rows]
col_lines = [ax.plot([], [], lw=1)[0] for _ in cols]
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

def init():
    scatter.set_offsets(np.empty((0, 2)))
    for line in row_lines + col_lines:
        line.set_data([], [])
    time_text.set_text("")
    return [scatter, *row_lines, *col_lines, time_text]

def animate(i):
    pts = positions[i]
    scatter.set_offsets(pts)
    # update rows and columns lines
    for idx, row in enumerate(rows):
        row_pts = pts[row]
        row_lines[idx].set_data(row_pts[:, 0], row_pts[:, 1])
    for idx, col in enumerate(cols):
        col_pts = pts[col]
        col_lines[idx].set_data(col_pts[:, 0], col_pts[:, 1])
    time_text.set_text(f"t = {time_vec[i]:.2f} s")
    return [scatter, *row_lines, *col_lines, time_text]

anim = FuncAnimation(fig, animate, frames=len(time_vec), init_func=init, blit=True, interval=40)

# -------------------------
# Render animation in Streamlit using JSHTML
# -------------------------
try:
    st.subheader("Animation")
    js_html = anim.to_jshtml()
    components.html(js_html, height=600)
    plt.close(fig)
except Exception as e:
    st.error("Animation rendering failed: " + str(e))
    # Fallback: show final configuration as static plot
    st.subheader("Final configuration (static fallback)")
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.set_aspect('equal', 'box')
    ax2.scatter(positions[-1, :, 0], positions[-1, :, 1], s=20)
    ax2.set_title("Final particle positions")
    st.pyplot(fig2)
    plt.close(fig2)

st.markdown("---")
st.caption("Notes: The app assumes particles are passive tracers moving with velocity u(x,y), v(x,y). "
           "For time-dependent velocity fields or variable density, Bernoulli-based pressure differences are not strictly valid.")
