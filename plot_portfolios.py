import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="Input parameters for plotting portfolio returns.")
    parser.add_argument("returns", help="Daily returns csv. Row is daily percentage returns, and columns are assets.")
    parser.add_argument("portfolio", help="Percentage of each portfolio. Each row is a portfolio, and each column are the percentage of each asset in each portfolio. Column must match with returns' columns.")
    parser.add_argument("output", help="Image output file.")
    args = parser.parse_args()
    return args

def cumulative_returns(data: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
    return (1.0 + data).cumprod(axis=axis) - 1.0

def main() -> None:
    args = get_args()
    returns = pd.read_csv(args.returns) # Shape = (days, assets)
    returns.set_index("date", inplace=True)
    returns.fillna(0, inplace=True)

    pf = pd.read_csv(args.portfolio) # Shape = (portfolios, assets)
    pf_returns = returns @ pf.T # Shape = (days, assets) @ (assets, portfolios) = (days, portfolios)
    pf_cumulative_returns = cumulative_returns(pf_returns, axis=1)

    plt.figure(figsize=(12, 8))
    for column in pf_cumulative_returns:
        plt.plot(pf_cumulative_returns.index, pf_cumulative_returns [column], label=column)

    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.output)
    plt.close()

if __name__ == "__main__":
    main()
