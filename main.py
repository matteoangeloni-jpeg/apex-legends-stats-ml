""""Apex Legends Stats Analysis - Main Script

This script performs comprehensive analysis of Apex Legends statistics including:
- Legends pick rates and meta analysis
- Weapons tier list and performance metrics
- Machine Learning predictions for player performance
- Interactive visualizations

Author: Matteo Angeloni
Date: March 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('dark_background')
sns.set_palette("husl")

print("\n" + "="*60)
print(" 📊 APEX LEGENDS - ML STATS ANALYSIS")
print("="*60 + "\n")
