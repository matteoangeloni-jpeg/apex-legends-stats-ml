# 🎮 apex-legends-stats-ml

> **Machine Learning analysis for Apex Legends** — legends pick rates, weapons tier list, player performance predictions using Random Forest and Gradient Boosting models.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Google Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00.svg?logo=googlecolab)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data](https://img.shields.io/badge/Data-32.3M%20players-red.svg)](https://apexlegendsstatus.com/)

---

## 📋 Overview

This project applies **Data Science** and **Machine Learning** to real Apex Legends statistics to:

- Analyze **live meta data** — pick rates, trends, tier lists for 27 legends and 30+ weapons
- Train **predictive models** — classify player performance tier and predict win rate
- Build a **recommendation engine** — suggest optimal legends and weapons based on player profile
- Generate **interactive dashboards** — 20+ professional visualizations with matplotlib/seaborn

**Data source:** [apexlegendsstatus.com](https://apexlegendsstatus.com) — 32.3 million active players, Season 25/26

---

## 📊 Datasets

### Legends (27 legends)
| Feature | Description |
|---|---|
| `pick_rate` | % of players using this legend (live) |
| `trend` | 7-day pick rate variation (%) |
| `avg_rank` | Average rank of players using this legend |
| `avg_level` | Average account level |
| `class` | Legend class (Skirmisher, Assault, Support, Recon, Controller) |

**Top 5 meta legends:** Octane (15.6%), Mad Maggie (8.0%), Valkyrie (7.7%), Bangalore (7.3%), Alter (6.7%)

### Weapons (30 weapons)
| Feature | Description |
|---|---|
| `dps_body` | Damage per second (body shots) |
| `ttk_purple_ms` | Time to Kill vs purple armor (ms) |
| `fire_rate` | Rounds per minute |
| `headshot_dmg` | Headshot damage |
| `tier` | Community tier (S+ → D) |
| `pick_rate` | % usage in ranked matches |

**Top S+ weapons:** Nemesis AR (18.2%), R-99 SMG (16.5%), 30-30 Repeater (11.7%)

### Synthetic Player Dataset (5,000 players)
Generated with realistic distributions based on real Apex Legends stats:

| Feature | Description |
|---|---|
| `kda` | Kill/Death/Assist ratio |
| `headshot_pct` | Headshot accuracy % |
| `survival_time` | Average survival time per match (seconds) |
| `damage_per_game` | Average damage dealt per match |
| `win_rate` | Win rate % |
| `games_played` | Total games played |
| `rank_num` | Rank (1=Bronze → 7=Predator) |
| `play_style` | Aggressivo / Tattico / Entry-Fragger / Difensivo / Support |
| `revives_per_game` | Average revives per match |

---

## 🤖 Machine Learning Models

### Model 1 — Random Forest Classifier (Performance Tier)
Predicts whether a player is **Low / Mid / High** performance tier.

```
Features:  KDA, Headshot%, SurvivalTime, Damage/Game, GamesPlayed,
           AccountLevel, Rank, Revives/Game, AssistRatio, PlayStyle, Legend
Target:    Performance Tier (Low=0, Mid=1, High=2)

Results:
  Accuracy (Test):       75.3%
  Cross-Val (5-fold):    74.3% ± 0.5%
  Top features:          Rank, Win Rate, KDA
```

### Model 2 — Gradient Boosting Regressor (Win Rate)
Predicts the player's **win rate** given their stats.

```
Features:  Same as above
Target:    Win Rate (%)

Results:
  MAE:    4.86%
  R²:     0.213
  CV R²:  0.192 ± 0.021
```

### Model 3 — Logistic Regression (Baseline)
```
  Accuracy (Test):       74.2%
  Cross-Val (5-fold):    74.3% ± 0.3%
```

### Model 4 — Random Forest (Weapons Tier Predictor)
Predicts optimal weapon tier from DPS, TTK, fire rate, pick rate.

```
  Leave-One-Out CV Accuracy:  70–80%
  Top predictive features:    TTK, DPS, pick_rate, fire_rate, hs_multiplier
```

---

## 🧩 Clustering Analysis (KMeans)

Unsupervised segmentation of 27 legends into **4 clusters** using PCA:

| Cluster | Label | Legends |
|---|---|---|
| 0 | **META** (High PR, Rising) | Octane, Mad Maggie, Valkyrie, Bangalore, Alter |
| 1 | **IN SALITA** (Low PR, Rising) | Revenant, Conduit, Crypto, Ballistic, Gibraltar |
| 2 | **NICHO** (Low PR, Declining) | Horizon, Loba, Wattson, Catalyst, Seer |
| 3 | **POPOLARE** (Mixed) | Lifeline, Sparrow, Wraith, Pathfinder, Fuse, Ash... |

---

## 🎯 Recommendation Systems

### Legend Recommender
Given a player profile, recommends the **top 3 legends** with match score, meta pick rate, and trend.

**Example — Aggressive player (Gold, KDA 2.3):**
```
1. Revenant    — Match Score: 25.7% | Meta: 4.3%  | Trend: +18.9%
2. Wraith      — Match Score: 19.4% | Meta: 4.6%  | Trend: -0.1%
3. Octane      — Match Score: 19.1% | Meta: 15.6% | Trend: +3.0%
Performance Tier: MID | Estimated Win Rate: 20.5%
```

### Weapon Recommender
Given play style + preferred range + headshot skill, recommends **top 5 weapons** with composite score.

**Example — Entry-Fragger, Close Range, High HS skill:**
```
# Weapon           Cat     Tier  DPS   TTK(ms)  Pick%   Score
1 R-99             SMG     S+    216   1050     16.5    0.892
2 Volt             SMG     S     195   1117     12.8    0.848
3 Nemesis          AR      S+    170   1050     18.2    0.835
4 P2020            Pistol  A     162   1100     14.2    0.798
5 CAR SMG          SMG     A     195   1100     8.3     0.771
```

---

## 📈 Visualizations

The notebook generates **4 dashboard PNG files**:

| File | Contents |
|---|---|
| `apex_meta_dashboard.png` | Top 15 pick rates, class distribution (pie), 7-day trends, scatter pick rate vs level |
| `apex_clustering.png` | PCA 2D cluster visualization, legend count per cluster |
| `apex_ml_results.png` | Feature importance, confusion matrix, win rate scatter, cross-validation boxplots |
| `apex_weapons_dashboard.png` | Top 15 weapons by pick rate, DPS vs TTK scatter, DPS per category, tier heatmap |

---

## 🔑 Key Insights

- **Meta weapons Season 25-26:** SMG dominance (R-99, Volt) + Marksman (30-30, Nemesis)
- **Top legends:** Octane, Mad Maggie, Valkyrie lead with >7% pick rate
- **Most predictive features for performance:** KDA, Play Style, and Rank
- **S+ tier weapons** have 2–3x pick rate compared to tier B–C
- **Win Rate correlates positively with KDA** (r ≈ 0.67)
- **Conduit** has the highest positive trend (+41.5%) among all legends

---

## 🗂️ Project Structure

```
apex-legends-stats-ml/
│
├── main.py                    # Main script — all analysis, models, dashboards
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .gitignore
│
├── (generated at runtime)
│   ├── apex_meta_dashboard.png
│   ├── apex_clustering.png
│   ├── apex_ml_results.png
│   └── apex_weapons_dashboard.png
```

---

## 🚀 How to Run

### Option 1 — Google Colab (recommended, no setup needed)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Open Google Colab
2. Upload `main.py` or paste the notebook cells
3. Click **Runtime → Run all**
4. Total runtime: ~2–3 minutes

### Option 2 — Local

```bash
# Clone the repository
git clone https://github.com/matteoangeloni-jpeg/apex-legends-stats-ml.git
cd apex-legends-stats-ml

# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

**Requirements:**
```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
scipy>=1.11
playwright>=1.40
```

---

## 📡 Data Sources

| Source | Description |
|---|---|
| [apexlegendsstatus.com](https://apexlegendsstatus.com/game-stats/legends-pick-rates) | Live pick rates — 32.3M players, Season 25/26 |
| [HFA Official TTK Data](https://www.reddit.com/r/apexlegends/) | Time-to-Kill values for all weapons Season 25 |
| [ComicBook Tier List](https://comicbook.com/gaming/news/apex-legends-weapon-tier-list-season-25/) | Community weapon tier list Season 25 |

---

## 🧰 Tech Stack

| Tool | Version | Usage |
|---|---|---|
| Python | 3.10+ | Core language |
| pandas | 2.0+ | Data manipulation |
| NumPy | 1.24+ | Numerical operations |
| scikit-learn | 1.3+ | ML models (RF, GB, LR, KMeans, PCA) |
| matplotlib | 3.7+ | Dashboards and charts |
| seaborn | 0.12+ | Statistical visualizations |
| Playwright | 1.40+ | Web scraping |
| Google Colab | — | Cloud GPU runtime |

---

## 📜 License

MIT License — Educational / Open Source project.

**Author:** Matteo Angeloni (March 2026)

> **Note:** Apex Legends data and trademarks © EA/Respawn Entertainment. This project is not affiliated with or endorsed by EA or Respawn Entertainment.
