# f1_race_points_prediction
Predicting whether an F1 driver will score points using qualifying and race data
# F1 Race Points Prediction

This project builds a machine learning model to predict whether a Formula 1 driver will score points in a race (finish in positions P1–P10) using pre-race information such as season, starting grid position, and qualifying position.

The project uses historical race, qualifying, and schedule data from a structured Formula 1 results dataset.

---

## 1. Techniques and Tools

- Language: Python  
- Libraries: `pandas`, `scikit-learn`  
- Model: Random Forest classifier  
- Methods:
  - Tabular data preprocessing and merging multiple CSV sources
  - Feature construction from race schedule, qualifying, and result data
  - Binary classification to predict whether a driver scores points
  - Evaluation with accuracy, F1-score, and a detailed classification report

---

## 2. Data and Features

The code assumes the following CSV files are available under a `data/` directory:

- `Race_Schedule.csv`  
- `Race_Results.csv`  
- `Qualifying_Results.csv`  

and that these files contain at least the columns:

- `raceId`, `year` (season/year)  
- `driverId`, `constructorId`  
- `grid` (starting grid position from race results)  
- `positionOrder` (final classified position)  
- `points` (race points scored)  
- `position` in `Qualifying_Results.csv` (qualifying position)

From these, the script constructs a modeling table and defines the target:

- `points_scored = 1` if `points > 0`  
- `points_scored = 0` otherwise  

Baseline features:

- season (`year`)  
- grid position (`grid`)  
- qualifying position (`quali_position` derived from qualifying results)

---

## 3. How to Run

1. Place the CSV files into a `data/` directory:

   ```text
   data/Race_Schedule.csv
   data/Race_Results.csv
   data/Qualifying_Results.csv
