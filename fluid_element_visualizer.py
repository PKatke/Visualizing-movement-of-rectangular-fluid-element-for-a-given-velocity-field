import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import matplotlib as mpl
mpl.rcParams['animation.writer'] = 'html'   # << IMPORTANT FIX

# Streamlit Page Title
st.title("Rectangular Fluid Element Visualizer (Steady, Inviscid, Constant Density)")

st.markdown("""
### Features:
1. **Checks flow assumptions** – Steady, inviscid, constant-density  
2. **Computes pressure difference** using simplified Bernoulli equation  
3. **Animates a rectangular fluid element** moving through the velocity field  
4. Runs **fully in Streamlit Cloud** (no ffmpeg required)
""")

# ================================
# USER INPUTS
# ================================
st.header("Velocity Field Definition")
u_str = st.text_input("u(x, y) =", "1 - 0.5*y")
v_str = st.text_input("v(x, y) =", "0.5*x")
rho_str = st.text_input("Density ρ =", "1")

st.header("Points for Pressure Difference")
x1 = st.number_input("x₁", value=0.0)
y1 = st.number_input("y₁", value=0.0)
x2 = st.number_input("x₂", value=1.0)
y2 = st.number_input("y₂", value=0.5)

st.header("Fluid Element Rectangle")
x0 = st.number_input("Rectangle start x", value=0.0)
y0 = st.number_input("Rectangle start y", value=0.0)
w = st.number_input("Width", value=0.4)
h = st.number_input("Height", value=0.2)
nx = st.number_input("Grid points in x", value=4)
ny = st.number_input("Grid points in y", value=4)

st.header("Animation Settings")
tmax = st.number_input("Simulation time", value=5.0)
dt = st.number_input("Time step", value=0.05)

# ================================
# PARSE EXPRESSIONS
# ================================
x, y = sp.symbols("x y")
u_expr = sp.sympify(u_str)
v_expr = sp.sympify(v_str)
rho_expr = sp.sympify(rho_str)

u = sp.lambdify((x, y), u_expr, "numpy")
v = sp.lambdify((x, y), v_expr, "numpy")
rho = sp.lambdify((x, y), rho_expr, "numpy")

# ================================
# ASSUMPTION CHECKS  
# ================================
st.subheader("Flow Assumption Checks")

steady = ("t" not in u_str) and ("t" not in v_str)
const_density = ("x" not in rho_str and "y" not in rho_str)

st.write("Steady flow:", "✔ Yes" if steady else "❌ No")
st.write("Constant density:", "✔ Yes" if const_density else "❌ No")
st.write("Inviscid flow:", "✔ Assumed")

if not steady:
    st.error("Velocity depends on time. Bernoulli cannot be used.")

if not const_density:
    st.error("Density varies with space. Bernoulli cannot be used.")

# ================================
# PRESSURE DIFFERENCE (BERNOULLI)
# ================================
if steady and const_density:
    st.subheader("Pressure Difference (Bernoulli)")

    u1 = u(x1, y1)
    v1 = v(x1, y1)

    u2 = u(x2, y2)
    v2 = v(x2, y2)

    rho_val = rho(0, 0)

    p1_minus_p2 = 0.5 * rho_val * ((u2**2 + v2**2) - (u1**2 + v1**2))

    st.success(f"Pressure difference p₁ - p₂ = **{p1_minus_p2:.5f}**")

# ================================
# PARTICLE ADVECTION
# ================================
st.header("Fluid Element Animation")

def velocity_field(pos, t):
    return [u(pos[0], pos[1]), v(pos[0], pos[1])]

# Create grid points of the rectangle
xs = np.linspace(x0, x0 + w, int(nx))
ys = np.linspace(y0, y0 + h, int(ny))
X0, Y0 = np.meshgrid(xs, ys)
points0 = np.stack([X0.flatten(), Y0.flatten()], axis=1)

# Time array
t_vals = np.arange(0, tmax, dt)

# Store trajectories
positions = np.zeros((len(t_vals), len(points0), 2))

# Integrate each particle
for i, p in enumerate(points0):
    sol = odeint(velocity_field, p, t_vals)
    positions[:, i, :] = sol

# ================================
# ANIMATION (NO FFMPEG)
# ================================
fig, ax = plt.subplots()
ax.set_xlim(min(X0.flatten()) - 1, max(X0.flatten()) + 1)
ax.set_ylim(min(Y0.flatten()) - 1, max(Y0.flatten()) + 1)
ax.set_aspect('equal')

scatter = ax.scatter([], [])

def update(frame):
    scatter.set_offsets(positions[frame])
    return scatter,

anim = FuncAnimation(fig, update, frames=len(t_vals), interval=50)

# Convert animation to HTML5 video (Base64, no ffmpeg needed)
video_html = anim.to_html5_video()

st.markdown("### Fluid Element Motion")
st.markdown(video_html, unsafe_allow_html=True)

plt.close(fig)

