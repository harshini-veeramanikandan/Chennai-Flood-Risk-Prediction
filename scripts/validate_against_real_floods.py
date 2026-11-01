"""
Validates the rainfall-proxy flood label against REAL, documented Chennai
flood events, using dates confirmed by news/Wikipedia sources:

  - 2015 South India floods: peak Dec 1, 2015 (494mm in 24h at Chennai airport)
  - 2021 South India floods: peak Nov 7 & Nov 11, 2021 (215mm / 210mm at Nungambakkam)
  - 2023 Cyclone Michaung floods: peak Dec 2-4, 2023 (530mm 3-day at Nungambakkam)

This checks: did our rainfall-intensity proxy actually flag these known
real flood days as high-risk? If yes, that's real evidence the proxy target
is meaningfully correlated with actual flooding, not just noise.
"""

import pandas as pd

FEATURES_PATH = "data/chennai_rainfall_features.csv"

REAL_FLOOD_EVENTS = [
    {"date": "2015-12-01", "event": "2015 South India floods (peak day)"},
    {"date": "2021-11-07", "event": "2021 South India floods (Nungambakkam 215mm)"},
    {"date": "2021-11-11", "event": "2021 South India floods (second peak)"},
    {"date": "2023-12-03", "event": "Cyclone Michaung"},
    {"date": "2023-12-04", "event": "Cyclone Michaung (Chennai airport closed)"},
]


def main():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    threshold = df["rainfall_3day_sum"].quantile(0.95)

    print(f"Flood-risk threshold (top 5% of 3-day rainfall): {threshold:.1f} mm\n")
    print(f"{'Date':<12} {'Event':<45} {'3-day rainfall':<16} {'Flagged as risk?'}")
    print("-" * 95)

    hits, total = 0, 0
    for event in REAL_FLOOD_EVENTS:
        row = df[df["Date"] == event["date"]]
        if row.empty:
            print(f"{event['date']:<12} {event['event']:<45} {'NOT IN DATASET':<16}")
            continue
        rainfall_3day = row["rainfall_3day_sum"].values[0]
        flagged = rainfall_3day >= threshold
        total += 1
        hits += int(flagged)
        print(f"{event['date']:<12} {event['event']:<45} {rainfall_3day:<16.1f} {'YES ✓' if flagged else 'NO ✗'}")

    print(f"\n{hits}/{total} known real flood days were correctly flagged by the rainfall-intensity proxy.")


if __name__ == "__main__":
    main()