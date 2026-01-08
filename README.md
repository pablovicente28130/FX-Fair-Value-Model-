# FX Fair Value Model

Un modèle de Fair Value pour le trading FX basé sur des fondamentaux macro-économiques, avec dashboard interactif.

## 📊 Aperçu

Ce projet implémente un modèle de Fair Value pour les devises, inspiré des approches institutionnelles. Il utilise des données macro-économiques pour estimer la "juste valeur" d'une devise et identifier les opportunités de trading.

![Dashboard Preview](docs/dashboard_preview.png)

## 🔧 Installation

```bash
# Cloner le repository
git clone https://github.com/[ton-username]/fx-fair-value-model.git
cd fx-fair-value-model

# Installer les dépendances
pip install -r requirements.txt
```

## 🚀 Utilisation

### Lancer le Dashboard

```bash
python dashboard.py data.xlsx
```

Puis ouvrir http://localhost:8050 dans un navigateur.

### Utiliser le Modèle en Python

```python
from model import FXFairValueAnalyzer, ModelConfig

# Configuration
config = ModelConfig(
    currencies=['EUR', 'CHF', 'CAD', 'CZK'],
    years_back=10,
    feature_xcats=['RYLDIRS02Y_NSA', 'CPIH_SA_P1M1ML12', 'INTRGDP_NSA_P1M1ML12_3MMA']
)

# Analyse
analyzer = FXFairValueAnalyzer('data.xlsx', config)
analyzer.load_data()
analyzer.analyze_all(window_months=12)

# Résultats
print(analyzer.get_ols_summary())
print(analyzer.get_significance_ratios())
```

## 📁 Structure du Projet

```
fx-fair-value-model/
├── model.py          # Logique du modèle (data loading, régression, fair value)
├── dashboard.py      # Dashboard Dash interactif
├── requirements.txt  # Dépendances Python
├── data.xlsx         # Données (non incluses, à fournir)
└── README.md         # Ce fichier
```

## 📈 Méthodologie

### Variables Explicatives

Le modèle utilise les facteurs macro suivants :

| Variable | Description |
|----------|-------------|
| `RYLDIRS02Y_NSA` | Taux réels 2 ans |
| `RYLDIRS05Y_NSA` | Taux réels 5 ans |
| `CPIH_SA_P1M1ML12` | Inflation headline YoY |
| `CPIC_SA_P1M1ML12` | Inflation core YoY |
| `INTRGDP_NSA_P1M1ML12_3MMA` | Croissance PIB YoY |

### Approche

1. **Régression OLS** : Estimation des sensibilités (betas) sur une fenêtre glissante
2. **Rolling OLS** : Capture de l'évolution temporelle des relations
3. **Fair Value** : Prédiction basée sur les facteurs macro
4. **Signal** : Écart entre le spot et la Fair Value

### Interprétation

- **Erreur positive** → Devise **surévaluée** → Signal de vente
- **Erreur négative** → Devise **sous-évaluée** → Signal d'achat

## 🖥️ Dashboard

Le dashboard interactif permet de :

- Visualiser les résultats des régressions OLS
- Explorer les rolling betas par devise ou par facteur
- Analyser les ratios de significativité
- Suivre les erreurs de Fair Value cumulées
- Comparer les signaux entre devises

### Fonctionnalités

- **Sélection de la fenêtre** : 6M, 9M, 12M, 18M, 24M
- **Graphiques interactifs** : zoom, hover, export
- **Tableaux formatés** : significativité avec \* et \*\*
- **Mise à jour en temps réel** : recalcul automatique

## 📊 Format des Données

Le modèle attend des données au format JPMaQS (format long) :

| real_date | cid | xcat | value |
|-----------|-----|------|-------|
| 2020-01-01 | EUR | RYLDIRS02Y_NSA | 0.5 |
| 2020-01-01 | EUR | CPIH_SA_P1M1ML12 | 1.2 |
| ... | ... | ... | ... |

### Devises supportées

EUR, CHF, CAD, CZK (et toute devise avec données `FXXR_NSA`)

## ⚠️ Limitations

1. **R² faible** : Normal pour les modèles FX à court terme
2. **Modèle linéaire** : Ne capture pas les non-linéarités
3. **Pas de coûts** : Le signal ne tient pas compte du spread/carry
4. **Mean-reversion** : Suppose un retour vers la Fair Value

## 🔬 Développement

### Ajouter des facteurs

Modifier `MODEL_CONFIG` dans `dashboard.py` :

```python
MODEL_CONFIG = ModelConfig(
    feature_xcats=[
        'RYLDIRS02Y_NSA',
        'CPIH_SA_P1M1ML12',
        'INTRGDP_NSA_P1M1ML12_3MMA',
        'NOUVEAU_FACTEUR',  # Ajouter ici
    ]
)
```

### Ajouter des devises

```python
MODEL_CONFIG = ModelConfig(
    currencies=['EUR', 'CHF', 'CAD', 'CZK', 'GBP'],  # Ajouter ici
)
```

## 📚 Références

- Inspiré de [FX Fair Value Model](https://github.com/NicolasHurbin/FX-Faire-Value-Model)
- Données : JPMaQS Quantamental Indicators

## 👤 Auteur

[Ton nom]

## 📄 License

MIT License
