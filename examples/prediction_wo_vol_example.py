import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from model import Kronos, KronosTokenizer, KronosPredictor


def plot_prediction(kline_df, pred_df, output_path="figures/prediction_wo_vol_example.png"):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df["close"]
    sr_pred_close = pred_df["close"]
    sr_close.name = "Ground Truth"
    sr_pred_close.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(close_df["Ground Truth"], label="Ground Truth", color="blue", linewidth=1.5)
    ax.plot(close_df["Prediction"], label="Prediction", color="red", linewidth=1.5)
    ax.set_ylabel("Close Price", fontsize=14)
    ax.legend(loc="lower left", fontsize=12)
    ax.grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

    data_path = os.path.join(os.path.dirname(__file__), "data", "XSHG_5min_600977.csv")
    df = pd.read_csv(data_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])

    lookback = 400
    pred_len = 120

    x_df = df.loc[:lookback-1, ["open", "high", "low", "close"]]
    x_timestamp = df.loc[:lookback-1, "timestamps"]
    y_timestamp = df.loc[lookback:lookback+pred_len-1, "timestamps"]

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    )

    print("Forecasted Data Head:")
    print(pred_df.head())

    kline_df = df.loc[:lookback+pred_len-1]
    plot_prediction(kline_df, pred_df)
