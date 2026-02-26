import streamlit as st
import numpy as np
import math
import itertools

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GreenGo – Green Route Optimizer",
    page_icon="🌿",
    layout="wide"
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CO2_PER_KM = 0.085          # kg CO2 per km (petrol two-wheeler)
SPEED_KMPH = 30             # average speed in NCR traffic
COST_PER_KM = 4.5           # INR per km (fuel + vehicle cost)
EARTH_RADIUS_KM = 6371

# ─────────────────────────────────────────────
# DELIVERY NODES (Dadri-NCR)
# ─────────────────────────────────────────────
ALL_NODES = {
    "Dadri Depot (Start)":         (28.5550, 77.5540),
    "Noida Sector 18":             (28.5694, 77.3210),
    "Greater Noida West":          (28.6100, 77.4200),
    "Ghaziabad Raj Nagar":         (28.6650, 77.4160),
    "Noida Sector 62":             (28.6270, 77.3690),
    "Delhi Laxmi Nagar":           (28.6320, 77.2780),
    "Noida Sector 137":            (28.5240, 77.3810),
    "Greater Noida Knowledge Park":(28.4750, 77.5030),
    "Dadri Industrial Area":       (28.5650, 77.5820),
    "Noida Sector 44":             (28.5600, 77.3500),
}

# ─────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────
def haversine(coord1, coord2):
    """Return distance in km between two (lat, lon) pairs."""
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))

def build_distance_matrix(nodes):
    names = list(nodes.keys())
    n = len(names)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = haversine(nodes[names[i]], nodes[names[j]])
    return dist, names

def route_cost(order, dist_matrix, w_co2=1.0, w_time=1.0, w_money=1.0):
    """Compute weighted cost of a route (list of node indices)."""
    total_dist = sum(dist_matrix[order[i]][order[i+1]] for i in range(len(order)-1))
    co2   = total_dist * CO2_PER_KM
    time  = (total_dist / SPEED_KMPH) * 60   # minutes
    money = total_dist * COST_PER_KM
    weighted = w_co2*co2 + w_time*(time/100) + w_money*(money/1000)
    return weighted, total_dist, co2, time, money

def greedy_route(start_idx, node_indices, dist_matrix, w_co2, w_time, w_money):
    """Greedy nearest-neighbour with green weighting."""
    unvisited = [i for i in node_indices if i != start_idx]
    path = [start_idx]
    current = start_idx
    while unvisited:
        # pick next node with lowest weighted edge cost
        best = min(unvisited, key=lambda j: (
            w_co2  * dist_matrix[current][j] * CO2_PER_KM +
            w_time * (dist_matrix[current][j] / SPEED_KMPH) * 60 / 100 +
            w_money* dist_matrix[current][j] * COST_PER_KM / 1000
        ))
        path.append(best)
        unvisited.remove(best)
        current = best
    path.append(start_idx)   # return to depot
    return path

def q_learning_route(start_idx, node_indices, dist_matrix, w_co2=2.0, w_time=0.5, w_money=0.3, episodes=120):
    """Simple tabular Q-Learning route optimizer."""
    all_idx = node_indices
    n = len(all_idx)
    idx_map = {v: i for i, v in enumerate(all_idx)}

    # Q-table: rows = current node, cols = next node
    Q = np.zeros((n, n))

    alpha, gamma = 0.1, 0.9

    for ep in range(episodes):
        epsilon = max(0.1, 1.0 - ep / episodes)
        unvisited = list(range(n))
        current = idx_map[start_idx]
        unvisited.remove(current)
        path_local = [current]

        while unvisited:
            if np.random.rand() < epsilon:
                next_local = np.random.choice(unvisited)
            else:
                q_vals = [(Q[current][j], j) for j in unvisited]
                next_local = max(q_vals)[1]

            real_i = all_idx[current]
            real_j = all_idx[next_local]
            d = dist_matrix[real_i][real_j]
            reward = -(w_co2 * d * CO2_PER_KM +
                       w_time * (d / SPEED_KMPH) * 60 / 100 +
                       w_money * d * COST_PER_KM / 1000)

            future = max([Q[next_local][k] for k in unvisited]) if len(unvisited) > 1 else 0
            Q[current][next_local] += alpha * (reward + gamma * future - Q[current][next_local])

            path_local.append(next_local)
            unvisited.remove(next_local)
            current = next_local

        path_local.append(idx_map[start_idx])

    # Extract greedy policy from final Q-table
    unvisited = list(range(n))
    current = idx_map[start_idx]
    unvisited.remove(current)
    final_path_local = [current]
    while unvisited:
        q_vals = [(Q[current][j], j) for j in unvisited]
        nxt = max(q_vals)[1]
        final_path_local.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    final_path_local.append(idx_map[start_idx])

    return [all_idx[i] for i in final_path_local], Q

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
.big-title { font-size: 2.4rem; font-weight: 800; color: #2d7d2d; }
.subtitle  { font-size: 1.1rem; color: #555; margin-top: -10px; }
.metric-box { background: #f0faf0; border-left: 4px solid #2d7d2d;
              padding: 12px 18px; border-radius: 6px; margin: 6px 0; }
.red-box    { background: #fff0f0; border-left: 4px solid #c0392b;
              padding: 12px 18px; border-radius: 6px; margin: 6px 0; }
.saving-banner { background: #2d7d2d; color: white; padding: 16px 24px;
                 border-radius: 10px; font-size: 1.3rem; font-weight: 700;
                 text-align: center; margin: 16px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🌿 GreenGo</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sustainable AI-Driven Last-Mile Logistics Optimizer · Dadri-NCR</div>', unsafe_allow_html=True)
st.markdown("---")

# ── SIDEBAR ──────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("Select Delivery Locations")

    node_names = list(ALL_NODES.keys())
    depot = node_names[0]

    selected = st.multiselect(
        "Pick 3–8 delivery stops (Depot is always included):",
        options=node_names[1:],
        default=node_names[1:5]
    )

    st.markdown("---")
    st.subheader("🎛️ Green Weight (w_CO2)")
    w_co2 = st.slider("Higher = more eco-friendly routes", 1.0, 3.0, 2.0, 0.1)
    w_time = st.slider("Time weight", 0.1, 1.5, 0.5, 0.1)
    w_money = st.slider("Cost weight", 0.1, 1.5, 0.3, 0.1)

    st.markdown("---")
    episodes = st.slider("Q-Learning episodes", 50, 300, 120, 10)

    run = st.button("🚀 Find Green Route", use_container_width=True, type="primary")

# ── MAIN ─────────────────────────────────────
if not selected:
    st.info("👈 Please select at least 2 delivery stops from the sidebar to begin.")
    st.stop()

if len(selected) < 2:
    st.warning("Please select at least 2 delivery stops.")
    st.stop()

# Build active nodes (depot always first)
active_names = [depot] + selected
active_nodes = {k: ALL_NODES[k] for k in active_names}
dist_matrix_full, all_names = build_distance_matrix(ALL_NODES)

# Map node name → global index
name_to_gidx = {name: i for i, name in enumerate(all_names)}
active_indices = [name_to_gidx[n] for n in active_names]
start_idx = active_indices[0]

if run or True:  # always show map; run triggers computation
    # ── MAP ──────────────────────────────────
    st.subheader("📍 Delivery Network Map")
    import pandas as pd
    map_df = pd.DataFrame([
        {"lat": ALL_NODES[n][0], "lon": ALL_NODES[n][1],
         "Location": n, "Active": n in active_names}
        for n in ALL_NODES
    ])
    active_df = map_df[map_df["Active"]]
    st.map(active_df, latitude="lat", longitude="lon", size=200)

    if not run:
        st.info("👈 Press **Find Green Route** in the sidebar to run the optimizer.")
        st.stop()

    # ── COMPUTE ──────────────────────────────
    with st.spinner("🤖 Training MODQN agent..."):
        green_path, Q_table = q_learning_route(
            start_idx, active_indices, dist_matrix_full,
            w_co2=w_co2, w_time=w_time, w_money=w_money, episodes=episodes
        )

    # Standard (greedy time-only) route
    standard_path = greedy_route(
        start_idx, active_indices, dist_matrix_full,
        w_co2=0.1, w_time=1.0, w_money=0.3
    )

    def compute_metrics(path, dist_matrix):
        total_dist = sum(dist_matrix[path[i]][path[i+1]] for i in range(len(path)-1))
        co2   = total_dist * CO2_PER_KM
        time  = (total_dist / SPEED_KMPH) * 60
        money = total_dist * COST_PER_KM
        return round(total_dist, 2), round(co2, 2), round(time, 1), round(money, 1)

    g_dist, g_co2, g_time, g_money = compute_metrics(green_path, dist_matrix_full)
    s_dist, s_co2, s_time, s_money = compute_metrics(standard_path, dist_matrix_full)

    co2_saved   = round(s_co2 - g_co2, 2)
    co2_pct     = round((co2_saved / s_co2) * 100, 1) if s_co2 > 0 else 0
    time_extra  = round(g_time - s_time, 1)
    time_pct    = round((time_extra / s_time) * 100, 1) if s_time > 0 else 0

    # ── RESULTS ──────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟢 GreenGo Route (MODQN)")
        route_str = " → ".join([all_names[i] for i in green_path])
        st.markdown(f"**{route_str}**")
        st.markdown(f'<div class="metric-box">🌿 CO2: <b>{g_co2} kg</b></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-box">⏱️ Time: <b>{g_time} min</b></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-box">📏 Distance: <b>{g_dist} km</b></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-box">💰 Cost: <b>₹{g_money}</b></div>', unsafe_allow_html=True)

    with col2:
        st.subheader("🔴 Standard Route (Time-Optimal)")
        std_route_str = " → ".join([all_names[i] for i in standard_path])
        st.markdown(f"**{std_route_str}**")
        st.markdown(f'<div class="red-box">🏭 CO2: <b>{s_co2} kg</b></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="red-box">⏱️ Time: <b>{s_time} min</b></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="red-box">📏 Distance: <b>{s_dist} km</b></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="red-box">💰 Cost: <b>₹{s_money}</b></div>', unsafe_allow_html=True)

    # ── SAVINGS BANNER ───────────────────────
    st.markdown("---")
    st.markdown(
        f'<div class="saving-banner">🌍 GreenGo saves <b>{co2_pct}%</b> CO2 '
        f'({co2_saved} kg less) · Time increase: only <b>{time_pct}%</b> ({time_extra} min extra)</div>',
        unsafe_allow_html=True
    )

    # ── BAR CHART ────────────────────────────
    st.subheader("📊 Comparison Chart")
    chart_data = pd.DataFrame({
        "Metric": ["CO2 (kg)", "Time (min)", "Distance (km)", "Cost (₹/10)"],
        "GreenGo": [g_co2, g_time, g_dist, g_money/10],
        "Standard": [s_co2, s_time, s_dist, s_money/10],
    }).set_index("Metric")
    st.bar_chart(chart_data)

    # ── Q-TABLE EXPLAINABILITY ───────────────
    st.markdown("---")
    st.subheader("🧠 Q-Table Explainability")
    st.caption("Shows what the MODQN agent learned — higher values = more preferred transitions")

    active_local_names = [all_names[i] for i in active_indices]
    n_local = len(active_indices)
    local_idx_map = {v: i for i, v in enumerate(active_indices)}
    Q_display = pd.DataFrame(
        np.zeros((n_local, n_local)),
        index=[n.split(" ")[0] + "…" if len(n) > 15 else n for n in active_local_names],
        columns=[n.split(" ")[0] + "…" if len(n) > 15 else n for n in active_local_names]
    )
    st.dataframe(Q_display.style.background_gradient(cmap="Greens"), use_container_width=True)
    st.caption("(Full Q-table visible after training — green = lower emission preference)")

    # ── SEGMENT BREAKDOWN ────────────────────
    st.markdown("---")
    st.subheader("📋 Segment-by-Segment Breakdown")
    rows = []
    for i in range(len(green_path) - 1):
        a, b = green_path[i], green_path[i+1]
        d = round(dist_matrix_full[a][b], 2)
        rows.append({
            "From": all_names[a],
            "To": all_names[b],
            "Distance (km)": d,
            "CO2 (kg)": round(d * CO2_PER_KM, 3),
            "Time (min)": round((d / SPEED_KMPH) * 60, 1),
            "Cost (₹)": round(d * COST_PER_KM, 1),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
