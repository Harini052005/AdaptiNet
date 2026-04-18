import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_data
from preprocessing import clean_data
from hypergraph import build_hypergraph_features
from rl_agent import QLearningAgent

st.set_page_config(page_title="Mini AdaptiNet Dashboard", layout="wide")

st.title("🚀 Mini AdaptiNet Dashboard")

# Load data
df = load_data("../data/dataset.csv")

# Clean data
df_clean = clean_data(df)

# Hypergraph features
grouped = build_hypergraph_features(df_clean)

# -------------------------------
# 📊 Raw vs Clean Data
# -------------------------------
st.header("📊 Raw vs Clean Engagement")

fig1, ax1 = plt.subplots()
ax1.plot(df["engagement"], label="Raw")
ax1.plot(df_clean["engagement"], label="Clean")
ax1.set_title("Raw vs Clean Engagement")
ax1.legend()

st.pyplot(fig1)

# -------------------------------
# 📈 Engagement by Time
# -------------------------------
st.header("📈 Engagement Trend by Time")

time_group = grouped.groupby("time")["engagement"].mean()

fig2, ax2 = plt.subplots()
ax2.bar(time_group.index, time_group.values)
ax2.set_xlabel("Time")
ax2.set_ylabel("Avg Engagement")
ax2.set_title("Average Engagement per Time")

st.pyplot(fig2)

# -------------------------------
# 🤖 RL Best Time
# -------------------------------
st.header("🤖 Best Time Prediction (Q-Learning)")

agent = QLearningAgent(n_states=4)
agent.train(grouped)

best_time = agent.best_time()

st.success(f"🔥 Best Time to Post: {best_time}")

# -------------------------------
# 📊 Q-table Visualization
# -------------------------------
st.header("📊 Q-Table Values")

q_values = agent.Q[:, 0]

fig3, ax3 = plt.subplots()
ax3.bar(range(len(q_values)), q_values)
ax3.set_xlabel("Time Slot")
ax3.set_ylabel("Q Value")
ax3.set_title("Learned Q-values")

st.pyplot(fig3)