# Netflix IMDb Mini Project (Extracted Script)

This is a cleaned-up Python script based on a data mining project around Netflix titles and their IMDb scores. It loads a Netflix dataset, does some quick exploration, runs a simple K-Means clustering, and includes a basic KNN section.

## What’s inside
- Load the CSV dataset with pandas
- Quick EDA (histograms, count plots, scatter plots)
- Handle missing values (fills missing age_certification)
- Correlation check (imdb_score vs imdb_votes)
- K-Means clustering using:
  - runtime
  - release_year
  - imdb_score
- Plotly visualizations (2D + 3D)
- A KNN training step (simple baseline)
- A custom “confusion-matrix style” evaluation based on a rule in the original project

## Requirements
You’ll need Python 3.9+ (3.8 is usually fine too) and these packages:

- numpy
- pandas
- matplotlib
- seaborn
- plotly
- scikit-learn

Install everything with:
```bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn
