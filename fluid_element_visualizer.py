import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import base64
import io
import matplotlib.animation as animation
FFMpegWriter = animation.FFMpegWriter

st.title("Rectangular Fluid Element Visualizer (Steady, Inviscid, Constant Density)")

st.markdown("""
This app:
1. Checks if the flow is **steady**, **inviscid**, and **constant-density**.  
2. Uses **Bernoulli’s equation** to compute pressure difference between two points.  
3. Animates a **rectangular fluid element** moving and deforming in the given velocity field.
""")

# -----------------------
# USER INPUTS
# -----------------------
st.header("Velocity field definitions")
u_str = st.text_input("u(x,y) =", "1 - 0.5*y")
v_str = st.text_input("v(x,y) =", "0.5*x")
rho_str = st.text_input("Density ρ =", "1")

st.header("Points for pressure difference")
x1 = st.number_input("x₁", value=0.0)
y1 = st.number_input("y₁", value=0.0)
x2 = st.number_input("x₂", value=1.0)
y2 = st.number_input("y₂", value=0.5)

st.header("Fluid element rectangle")
x0 = st.number_input("Rect start x", value=0.0)
y0 = st.number_input("Rect start y", value=0.0)
width = st.number_input("Width", value=0.4)
height = st.number_input("Height", value=0.2)
nx = st.number_input("Grid points (x-direction)", value=4)
ny = st.number_input("Grid points (y-direction)", value=4)

st.header("Animation settings")
tmax = st.number_input("Simulation time", value=5.0)
dt = st.number_input("Time step", value=0.05)


# -----------------------------------
# PARSE SYMBOLIC EXPRESSIONS
# -----------------------------------
x, y = sp.symbols("x y")
u_expr = sp.sympify(u_str)
v_expr = sp.sympify(v_str)
rho_expr = sp.sympify(rho_str)

# Convert to lambda
u = sp.lambdify((x, y), u_expr, "numpy")
v = sp.lambdify((x, y), v_expr, "numpy")
rho = sp.lambdify((x, y), rho_expr, "numpy")

# -----------------------------------
# CHECK ASSUMPTIONS
# -----------------------------------
st.subheader("Flow Assumption Checks")

steady = "t" not in u_str and "t" not in v_str
const_density = rho_str.replace(" ", "") not in ["x", "y"] and ("x" not in rho_str and "y" not in rho_str)

st.write("Steady flow:", "Yes" if steady else "No")
st.write("Constant density:", "Yes" if const_density else "No")
st.write("Inviscid flow:", "Assumed")

if not steady:
    st.error("Velocity field contains time — Bernoulli cannot be applied.")
if not const_density:
    st.error("Density varies — Bernoulli cannot be applied.")


# -----------------------------------
# PRESSURE DIFFERENCE USING BERNOULLI
# -----------------------------------
if steady and const_density:
    st.subheader("Pressure Difference Using Bernoulli")

    u1 = u(x1, y1)
    v1 = v(x1, y1)

    u2 = u(x2, y2)
    v2 = v(x2, y2)

    rho_val = rho(0, 0)

    p1_minus_p2 = 0.5 * rho_val * ((u2**2 + v2**2) - (u1**2 + v1**2))

    st.success(f"Pressure difference p₁ - p₂ = **{p1_minus_p2:.5f}**")


# -----------------------------------
# ADVECTION OF FLUID ELEMENT
# -----------------------------------
st.header("Animation Output")

def velocity_field(pos, t):
    return [u(pos[0], pos[1]), v(pos[0], pos[1])]

# Grid points
xs = np.linspace(x0, x0 + width, int(nx))
ys = np.linspace(y0, y0 + height, int(ny))
X0, Y0 = np.meshgrid(xs, ys)
points0 = np.stack([X0.flatten(), Y0.flatten()], axis=1)

# Integrate particle trajectories
t_vals = np.arange(0, tmax, dt)
positions = np.zeros((len(t_vals), len(points0), 2))

for i, p in enumerate(points0):
    sol = odeint(velocity_field, p, t_vals)
    positions[:, i, :] = sol

# -----------------------------------
# CREATE ANIMATION
# -----------------------------------
fig, ax = plt.subplots()
ax.set_xlim(min(X0.flatten()) - 1, max(X0.flatten()) + 1)
ax.set_ylim(min(Y0.flatten()) - 1, max(Y0.flatten()) + 1)
ax.set_aspect('equal')
scatter = ax.scatter([], [])

def update(frame):
    scatter.set_offsets(positions[frame])
    return scatter,

anim = FuncAnimation(fig, update, frames=len(t_vals), interval=50)

# -----------------------------------
# FIX: Save using FFmpegWriter
# -----------------------------------
import tempfile
import os
import matplotlib.animation as animation

FFMpegWriter = animation.FFMpegWriter  # Force use

temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
temp_video_path = temp_video.name
temp_video.close()

writer = FFMpegWriter(fps=20)
anim.save(temp_video_path, writer=writer)

# Read file and convert
with open(temp_video_path, "rb") as f:
    video_bytes = f.read()

video_b64 = base64.b64encode(video_bytes).decode()

st.markdown(
    f"<video controls><source src='data:video/mp4;base64,{video_b64}' type='video/mp4'></video>",
    unsafe_allow_html=True,
)

os.remove(temp_video_path)
