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


def plot_prediction(kline_df, pred_df, output_path="figures/prediction_batch_example.png"):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df["close"]
    sr_pred_close = pred_df["close"]
    sr_close.name = "Ground Truth"
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df["volume"]
    sr_pred_volume = pred_df["volume"]
    sr_volume.name = "Ground Truth"
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(close_df["Ground Truth"], label="Ground Truth", color="blue", linewidth=1.5)
    ax1.plot(close_df["Prediction"], label="Prediction", color="red", linewidth=1.5)
    ax1.set_ylabel("Close Price", fontsize=14)
    ax1.legend(loc="lower left", fontsize=12)
    ax1.grid(True)

    ax2.plot(volume_df["Ground Truth"], label="Ground Truth", color="blue", linewidth=1.5)
    ax2.plot(volume_df["Prediction"], label="Prediction", color="red", linewidth=1.5)
    ax2.set_ylabel("Volume", fontsize=14)
    ax2.legend(loc="upper left", fontsize=12)
    ax2.grid(True)

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

    dfs = []
    xtsp = []
    ytsp = []
    for i in range(5):
        start = i * 20
        idf = df.loc[start:start+lookback-1, ["open", "high", "low", "close", "volume", "amount"]]
        i_x_timestamp = df.loc[start:start+lookback-1, "timestamps"]
        i_y_timestamp = df.loc[start+lookback:start+lookback+pred_len-1, "timestamps"]

        dfs.append(idf)
        xtsp.append(i_x_timestamp)
        ytsp.append(i_y_timestamp)

    pred_df = predictor.predict_batch(
        df_list=dfs,
        x_timestamp_list=xtsp,
        y_timestamp_list=ytsp,
        pred_len=pred_len,
    )

    print("Batch prediction complete. Number of results:", len(pred_df))
    for i, p in enumerate(pred_df[:1]):
        print(f"Result {i} head:")
        print(p.head())

    kline_df = df.loc[:lookback+pred_len-1]
    plot_prediction(kline_df, pred_df[0])
