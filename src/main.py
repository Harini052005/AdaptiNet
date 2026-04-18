from data_loader import load_data
from preprocessing import clean_data
from hypergraph import build_hypergraph_features
from model import train_model
from rl_agent import QLearningAgent
import pandas as pd

def main():
    # Load data
    df = load_data("../data/dataset.csv")

    # Clean data
    df_clean = clean_data(df)

    # Build hypergraph-like features
    grouped = build_hypergraph_features(df_clean)

    # Train ML model
    model = train_model(grouped)

    # Prediction (fix warning also)
    prediction = model.predict(pd.DataFrame([[2]], columns=["time"]))
    print("Predicted engagement at time=2:", prediction)

    # RL training
    agent = QLearningAgent(n_states=4)
    agent.train(grouped)

    print("Best time to post (Q-learning):", agent.best_time())

if __name__ == "__main__":
    main()