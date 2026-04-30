import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

def run_eda_and_hypothesis():
    print("Loading enriched dataset...")
    df = pd.read_csv('nba_games_enriched.csv')
    df['Win'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)
    
    # --- GLOBAL STYLE SETTINGS ---
    # Daha profesyonel, modern bir görünüm için stil ayarları
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold'
    })
    
    print("Generating Professional EDA plots...")
    
    # --- EDA 1: Win Rate by Days of Rest ---
    plt.figure(figsize=(11, 6))
    rest_win_rates = df.groupby('Days_of_Rest')['Win'].mean().reset_index()
    rest_win_rates = rest_win_rates[rest_win_rates['Days_of_Rest'] <= 5]
    
    # Renk paleti ve tasarım iyileştirmesi (FutureWarning düzeltildi)
    ax = sns.barplot(
        x='Days_of_Rest', 
        y='Win', 
        data=rest_win_rates, 
        hue='Days_of_Rest', 
        palette='coolwarm', 
        legend=False,
        edgecolor='black',
        linewidth=1.5
    )
    
    plt.title('Impact of Rest on Team Win Rate', fontsize=16, pad=20)
    plt.xlabel('Days of Rest (0 = Back-to-Back)', fontsize=13)
    plt.ylabel('Win Probability', fontsize=13)
    
    # Lig ortalaması çizgisi
    league_avg = df['Win'].mean()
    plt.axhline(league_avg, color='#e74c3c', linestyle='--', linewidth=2, label=f'League Avg ({league_avg:.3f})')
    
    # Çubukların üzerine değerleri (yüzdeleri) ekleme
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', 
                    fontsize=11, color='black', xytext=(0, 5), 
                    textcoords='offset points')
        
    plt.ylim(0, 0.7) # Y eksenini daha orantılı göster
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('eda_rest_vs_win.png', dpi=300) # Yüksek çözünürlüklü kayıt
    plt.close()
    
    # --- EDA 2: Distribution of Distance Traveled ---
    plt.figure(figsize=(11, 6))
    
    # Şık bir dağılım grafiği (histogram + kde)
    sns.histplot(
        df[df['Distance_Traveled_km'] > 0]['Distance_Traveled_km'], 
        bins=40, 
        kde=True, 
        color='#2980b9', 
        edgecolor='white',
        linewidth=1.2,
        alpha=0.7
    )
    
    plt.title('Distribution of Pre-Game Flight Distances', fontsize=16, pad=20)
    plt.xlabel('Distance Traveled (km)', fontsize=13)
    plt.ylabel('Frequency (Number of Games)', fontsize=13)
    plt.tight_layout()
    plt.savefig('eda_distance_distribution.png', dpi=300)
    plt.close()

    print("High-resolution plots saved as 'eda_rest_vs_win.png' and 'eda_distance_distribution.png'.")
    
    # --- HYPOTHESIS TESTING: Denver Altitude Effect ---
    print("\n" + "="*60)
    print("HYPOTHESIS TESTING: THE DENVER ALTITUDE EFFECT")
    print("="*60)
    
    away_teams = df[df['Is_Home'] == 0]
    denver_away_games = away_teams[away_teams['Location_Abbr'] == 'DEN']['PTS']
    
    sea_level_teams = ['MIA', 'BKN', 'NYK', 'LAL', 'LAC']
    sea_level_away_games = away_teams[away_teams['Location_Abbr'].isin(sea_level_teams)]['PTS']
    
    print(f"Average Away Points Scored in Denver (High Altitude): {denver_away_games.mean():.2f}")
    print(f"Average Away Points Scored at Sea Level:              {sea_level_away_games.mean():.2f}")
    
    t_stat, p_value = stats.ttest_ind(denver_away_games, sea_level_away_games, equal_var=False)
    
    print(f"\nIndependent T-Test Results:")
    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value:     {p_value:.4f}")
    
    print("\nCONCLUSION:")
    if p_value < 0.05:
        print("P-value < 0.05. The Denver altitude has a STATISTICALLY SIGNIFICANT negative impact on away team scoring. Hypothesis is SUPPORTED!")
    else:
        print("P-value >= 0.05. We cannot prove a statistically significant effect of Denver altitude on scoring. Other factors may dominate.")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_eda_and_hypothesis()