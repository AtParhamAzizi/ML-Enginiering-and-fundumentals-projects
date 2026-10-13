"""Create a friendly visual summary of the Ames housing data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter


def money(value: float, _position: int) -> str:
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value / 1_000:.0f}k"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("datasets/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/house_price_story.png"))
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    required = {"SalePrice", "GrLivArea", "OverallQual", "Neighborhood", "YearBuilt"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")

    plt.style.use("seaborn-v0_8-whitegrid")
    navy = "#17324D"
    blue = "#2F80ED"
    coral = "#E76F51"
    gold = "#E9C46A"
    green = "#2A9D8F"

    figure, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor="#F7F9FC")
    figure.suptitle(
        "What shapes a home’s sale price?",
        fontsize=25,
        fontweight="bold",
        color=navy,
        x=0.06,
        ha="left",
    )
    figure.text(
        0.06,
        0.925,
        "A visual tour of the Ames housing market — every point is a real home in the training data.",
        fontsize=12,
        color="#52606D",
    )

    price_axis = axes[0, 0]
    price_axis.hist(data["SalePrice"], bins=35, color=blue, alpha=0.9, edgecolor="white")
    price_axis.axvline(data["SalePrice"].median(), color=coral, linewidth=2, label="Median")
    price_axis.set_title("Most homes cluster below $300k", loc="left", fontweight="bold", color=navy)
    price_axis.set_xlabel("Sale price")
    price_axis.set_ylabel("Number of homes")
    price_axis.xaxis.set_major_formatter(FuncFormatter(money))
    price_axis.legend(frameon=False)

    neighborhood_axis = axes[0, 1]
    neighborhood = data.groupby("Neighborhood")["SalePrice"].median().sort_values().tail(10)
    neighborhood_axis.barh(neighborhood.index, neighborhood.values, color=green)
    neighborhood_axis.set_title("Where you live matters", loc="left", fontweight="bold", color=navy)
    neighborhood_axis.set_xlabel("Median sale price")
    neighborhood_axis.xaxis.set_major_formatter(FuncFormatter(money))
    neighborhood_axis.tick_params(axis="y", labelsize=8)

    quality_axis = axes[0, 2]
    quality = data.groupby("OverallQual")["SalePrice"].median()
    quality_axis.plot(quality.index, quality.values, marker="o", linewidth=3, color=coral)
    quality_axis.fill_between(quality.index, quality.values, alpha=0.12, color=coral)
    quality_axis.set_title("Quality has a clear payoff", loc="left", fontweight="bold", color=navy)
    quality_axis.set_xlabel("Overall quality score")
    quality_axis.set_ylabel("Median sale price")
    quality_axis.yaxis.set_major_formatter(FuncFormatter(money))
    quality_axis.set_xticks(quality.index)

    area_axis = axes[1, 0]
    area_axis.scatter(data["GrLivArea"], data["SalePrice"], s=18, alpha=0.35, color=blue, edgecolors="none")
    area_axis.set_title("More living space usually costs more", loc="left", fontweight="bold", color=navy)
    area_axis.set_xlabel("Above-ground living area (sq ft)")
    area_axis.set_ylabel("Sale price")
    area_axis.yaxis.set_major_formatter(FuncFormatter(money))

    year_axis = axes[1, 1]
    yearly = data.groupby("YearBuilt")["SalePrice"].median().rolling(5, min_periods=1).mean()
    year_axis.plot(yearly.index, yearly.values, color=gold, linewidth=3)
    year_axis.set_title("Newer homes tend to command more", loc="left", fontweight="bold", color=navy)
    year_axis.set_xlabel("Year built")
    year_axis.set_ylabel("Smoothed median price")
    year_axis.yaxis.set_major_formatter(FuncFormatter(money))

    takeaway_axis = axes[1, 2]
    takeaway_axis.axis("off")
    takeaway_axis.text(0.02, 0.90, "A few takeaways", fontsize=16, fontweight="bold", color=navy)
    takeaways = [
        f"• Typical home: {money(data['SalePrice'].median(), 0)}",
        f"• Middle living area: {data['GrLivArea'].median():,.0f} sq ft",
        f"• Most common quality score: {int(data['OverallQual'].mode().iloc[0])}/10",
        "• Neighborhood and quality move prices the most",
    ]
    takeaway_axis.text(0.02, 0.70, "\n\n".join(takeaways), fontsize=12, color="#334E68", va="top", linespacing=1.5)
    takeaway_axis.text(0.02, 0.10, "Source: datasets/train.csv", fontsize=9, color="#829AB1")

    for axis in axes.flat:
        axis.set_facecolor("#FFFFFF")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)

    figure.tight_layout(rect=(0, 0, 1, 0.90))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
