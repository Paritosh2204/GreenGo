import streamlit as st
import numpy as np
import math
import pandas as pd

st.set_page_config(page_title="GreenGo – Green Route Optimizer", page_icon="🌿", layout="wide")

CO2_PER_KM   = 0.085
SPEED_KMPH   = 30
COST_PER_KM  = 4.5
EARTH_RADIUS = 6371

ALL_NODES = {
    "Dadri Depot (Start)":          (28.5550, 77.5540),
    "Noida Sector 18":              (28.5694, 77.3210),
    "Greater Noida West":           (28.6100, 77.4200),
    "Ghaziabad Raj Nagar":          (28.6650, 77.4160),
    "Noida Sector 62":              (28.6270, 77.3690),
    "Delhi Laxmi Nagar":            (28.6320, 77.2780),
    "Noida Sector 137":             (28.5240, 77.3810),
    "Greater Noida Knowledge Park": (28.4750, 77.5030),
    "Dadri Industrial Area":        (28.5650, 77.5820),
    "Noida Sector 44":              (28.5600, 77.3500),
}

def haversine(c1, c2):
    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 2 * EARTH_RADIUS * math.asin(math.sqrt(a))

def build_dist_matrix(node_dict):
    names = list(node_dict.keys())
    n = len(names)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i][j] = haversine(node_dict[names[i]], node_dict[names[j]])
    return D, names

def get_metrics(path, D):
    dist  = sum(D[path[i]][path[i+1]] for i in range(len(path)-1))
    co2   = round(dist * CO2_PER_KM, 2)
    time  = round((dist / SPEED_KMPH) * 60, 1)
    money = round(dist * COST_PER_KM, 1)
    return round(dist, 2), co2, time, money

def standard_route(start, indices, D):
    unvisited = [i for i in indices if i != start]
    path, cur = [start], start
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[cur][j])
        path.append(nxt); unvisited.remove(nxt); cur = nxt
    path.append(start)
    return path

def green_route_ql(start, indices, D, w_co2=2.0, w_time=0.5, w_money=0.3, episodes=150):
    n   = len(indices)
    idx = {v: i for i, v in enumerate(indices)}
    rev = {i: v for v, i in idx.items()}
    Q   = np.zeros((n, n))
    alpha, gamma = 0.15, 0.9

    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / episodes)
        cur = idx[start]
        unvisited = [i for i in range(n) if i != cur]
        while unvisited:
            if np.random.rand() < eps:
                nxt = np.random.choice(unvisited)
            else:
                nxt = max(unvisited, key=lambda j: Q[cur][j])
            gi, gj = rev[cur], rev[nxt]
            d = D[gi][gj]
            reward = -(w_co2 * d * CO2_PER_KM + w_time * (d/SPEED_KMPH)*60/100 + w_money * d * COST_PER_KM/1000)
            future = max(Q[nxt][k] for k in unvisited if k != nxt) if len(unvisited) > 1 else 0
            Q[cur][nxt] += alpha * (reward + gamma * future - Q[cur][nxt])
            unvisited.remove(nxt); cur = nxt

    cur = idx[start]
    unvisited = [i for i in range(n) if i != cur]
    path_local = [cur]
    while unvisited:
        nxt = max(unvisited, key=lambda j: Q[cur][j])
        path_local.append(nxt); unvisited.remove(nxt); cur = nxt
    path_local.append(idx[start])
    return [rev[i] for i in path_local], Q, [rev[i] for i in range(n)]

# ── STYLES ──
st.markdown("""
<style>
.big-title  { font-size:2.4rem; font-weight:800; color:#2d7d2d; }
.subtitle   { font-size:1.1rem; color:#888; margin-top:-10px; }
.green-box  { background:#d4edda; border-left:5px solid #2d7d2d; padding:12px 18px;
              border-radius:6px; margin:6px 0; color:#155724; font-weight:600; font-size:1rem; }
.red-box    { background:#f8d7da; border-left:5px solid #c0392b; padding:12px 18px;
              border-radius:6px; margin:6px 0; color:#721c24; font-weight:600; font-size:1rem; }
.saving-ok  { background:#2d7d2d; color:#ffffff; padding:16px 24px; border-radius:10px;
              font-size:1.25rem; font-weight:700; text-align:center; margin:16px 0; }
.saving-zero{ background:#e67e22; color:#ffffff; padding:16px 24px; border-radius:10px;
              font-size:1.1rem; font-weight:700; text-align:center; margin:16px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🌿 GreenGo</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sustainable AI-Driven Last-Mile Logistics Optimizer · Dadri-NCR</div>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuration")
    node_names = list(ALL_NODES.keys())
    depot = node_names[0]
    selected = st.multiselect("Pick 3–8 delivery stops:", options=node_names[1:], default=node_names[1:5])
    st.markdown("---")
    st.subheader("🎛️ Green Weight (w_CO2)")
    w_co2   = st.slider("CO2 weight", 1.0, 3.0, 2.0, 0.1)
    w_time  = st.slider("Time weight", 0.1, 1.5, 0.5, 0.1)
    w_money = st.slider("Cost weight", 0.1, 1.5, 0.3, 0.1)
    episodes= st.slider("Q-Learning episodes", 50, 300, 150, 10)
    run = st.button("🚀 Find Green Route", use_container_width=True, type="primary")

if len(selected) < 2:
    st.info("👈 Please select at least 2 delivery stops from the sidebar.")
    st.stop()

active_names = [depot] + selected
st.subheader("📍 Delivery Network Map")
map_df = pd.DataFrame([{"lat": ALL_NODES[n][0], "lon": ALL_NODES[n][1]} for n in active_names])
st.map(map_df, latitude="lat", longitude="lon", size=300)

if not run:
    st.info("👈 Press **Find Green Route** in the sidebar to run the optimizer.")
    st.stop()

active_nodes = {k: ALL_NODES[k] for k in active_names}
D, local_names = build_dist_matrix(active_nodes)
n = len(local_names)
all_indices = list(range(n))

with st.spinner("🤖 Training MODQN agent..."):
    g_path, Q, q_names = green_route_ql(0, all_indices, D, w_co2=w_co2, w_time=w_time, w_money=w_money, episodes=episodes)

s_path = standard_route(0, all_indices, D)

g_dist, g_co2, g_time, g_money = get_metrics(g_path, D)
s_dist, s_co2, s_time, s_money = get_metrics(s_path, D)

co2_saved = round(s_co2 - g_co2, 2)
co2_pct   = round((co2_saved / s_co2) * 100, 1) if s_co2 > 0 else 0
time_diff = round(g_time - s_time, 1)
time_pct  = round((abs(time_diff) / s_time) * 100, 1) if s_time > 0 else 0

def rstr(path): return " → ".join(local_names[i] for i in path)

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("🟢 GreenGo Route (MODQN)")
    st.write(rstr(g_path))
    st.markdown(f'<div class="green-box">🌿 CO2: {g_co2} kg</div>',     unsafe_allow_html=True)
    st.markdown(f'<div class="green-box">⏱️ Time: {g_time} min</div>',   unsafe_allow_html=True)
    st.markdown(f'<div class="green-box">📏 Distance: {g_dist} km</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="green-box">💰 Cost: ₹{g_money}</div>',     unsafe_allow_html=True)

with col2:
    st.subheader("🔴 Standard Route (Time-Optimal)")
    st.write(rstr(s_path))
    st.markdown(f'<div class="red-box">🏭 CO2: {s_co2} kg</div>',      unsafe_allow_html=True)
    st.markdown(f'<div class="red-box">⏱️ Time: {s_time} min</div>',    unsafe_allow_html=True)
    st.markdown(f'<div class="red-box">📏 Distance: {s_dist} km</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="red-box">💰 Cost: ₹{s_money}</div>',      unsafe_allow_html=True)

st.markdown("---")
if co2_pct > 0:
    st.markdown(f'<div class="saving-ok">🌍 GreenGo saves <b>{co2_pct}%</b> CO2 ({co2_saved} kg less) · Time change: <b>{time_pct}%</b> ({time_diff} min)</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="saving-zero">⚠️ Same path found. Try more stops or increase CO2 weight slider above 2.0!</div>', unsafe_allow_html=True)

st.subheader("📊 Comparison Chart")
chart_df = pd.DataFrame({
    "GreenGo":  [g_co2, g_time, g_dist, g_money/10],
    "Standard": [s_co2, s_time, s_dist, s_money/10],
}, index=["CO2 (kg)", "Time (min)", "Distance (km)", "Cost (₹/10)"])
st.bar_chart(chart_df)

st.markdown("---")
st.subheader("🧠 Q-Table Explainability")
st.caption("Learned Q-values — more negative = agent avoids that edge (high emission). Less negative = preferred green path.")
short = [nm[:12]+"…" if len(nm)>12 else nm for nm in local_names]
Q_df  = pd.DataFrame(Q.round(4), index=short, columns=short).reset_index().rename(columns={"index":"Location"})
st.dataframe(Q_df.astype(str), use_container_width=True)
st.caption(f"Rows = current location · Columns = next location · Trained over {episodes} episodes")

st.markdown("---")
st.subheader("📋 Segment-by-Segment Breakdown (Green Route)")
rows = []
for i in range(len(g_path)-1):
    a, b = g_path[i], g_path[i+1]
    d = round(D[a][b], 2)
    rows.append({"From": local_names[a], "To": local_names[b],
                 "Distance (km)": d, "CO2 (kg)": round(d*CO2_PER_KM,3),
                 "Time (min)": round((d/SPEED_KMPH)*60,1), "Cost (₹)": round(d*COST_PER_KM,1)})
st.dataframe(pd.DataFrame(rows), use_container_width=True)
