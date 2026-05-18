# 🏀 NBA Performance Analysis: Quantifying Travel, Rest, and Altitude

**DSA 210 – Introduction to Data Science (Spring 2025-2026)** **Student:** Sarp Süer  
**Institution:** Sabancı University  
**Project Status:** 🟢 Completed (Data Collection, EDA, Hypothesis Testing, Machine Learning and Final Report)

---

## 📌 Motivation
In the NBA, the concept of a "schedule loss" is widely debated. An 82-game regular season involves brutal travel schedules, back-to-back games, and changing time zones. This project aims to **systematically quantify these factors using data science**. 

By combining game-level box scores with geographic API data, I evaluate how schedule congestion (Days of Rest), flight fatigue (Distance Traveled), and topographical factors (Altitude) impact a team's win probability and offensive output.

---

## 📊 Exploratory Data Analysis (EDA)

Before running statistical tests, I analyzed the core distributions of our variables. Below are the key visual insights:

### 1. Win Rate vs. Days of Rest
Does extra rest actually help you win? We analyzed the win percentages based on the number of days between games.

![Win Rate by Rest Days](eda_rest_vs_win.png)
> **Observation:** There is a noticeable drop in win probability for teams on "Zero" days of rest (Back-to-Backs).

### 2. Travel Distance Distribution
The NBA is a league of extremes. We mapped out how many kilometers teams typically travel between away games.

![Distance Distribution](eda_distance_distribution.png)
> **Observation:** While most travel is regional, a significant portion of the schedule involves cross-country flights exceeding 2000km.

---

## 🔬 Hypothesis Testing Results

I conducted four independent t-tests to validate the "Schedule Loss" theory numerically. 

### Hypothesis 1: The Back-to-Back (B2B) Fatigue Effect
* **H1:** Teams playing on a back-to-back score significantly fewer points.
* **Result:** 🔴 **Reject H0 (Significant)**
* **Conclusion:** Fatigue reduces offensive output measurably.

### Hypothesis 2: The "Cross-Country" Flight Penalty
* **H1:** Away teams traveling >1500 km score fewer points than those traveling <500 km.
* **Result:** 🔴 **Reject H0 (Significant)**
* **Conclusion:** Long-distance travel acts as a significant "tax" on team performance.

### Hypothesis 3: The Denver Altitude Effect
* **H1:** Away teams score significantly fewer points in Denver (1609m) vs. sea-level cities.
* **Result:** 🟢 **Fail to reject H0 (Not Significant)**
* **Conclusion:** Surprisingly, when controlling for other factors, altitude did not show a statistically significant impact on points in our current sample (p > 0.05).

### Hypothesis 4: The "Rust vs. Rest" Dilemma
* **H1:** Teams with extended rest (3+ days) score differently than standard rest (1-2 days).
* **Result:** 🟡 **Fail to reject H0 (p=0.052)**
* **Conclusion:** While very close to the significance threshold, we cannot definitively say that "rust" exists, though a slight trend is visible.

---

## 🤖 Machine Learning Models
In the final phase, the following features were used to predict game outcomes:
- `Distance_Traveled_km`
- `Days_of_Rest`
- `Altitude_Difference`

**Models & Results:** Logistic Regression and Random Forest Classifiers were trained. The Random Forest model achieved the highest predictive accuracy at **54.13%**. Feature importance analysis revealed that travel distance and altitude differences held the highest predictive weights.
---

## ⚙️ Setup and Reproducibility

### Installation
```bash
git clone [https://github.com/yourusername/NBA-Performance-Analysis.git](https://github.com/yourusername/NBA-Performance-Analysis.git)
cd NBA-Performance-Analysis
pip install -r requirements.txt
