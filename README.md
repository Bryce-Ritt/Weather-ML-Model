# Weather-ML-Model v1.1

Overview

This project utilizes stacked machine learning models to achieve hourly temperature forecasting for the coming week in a given US location (based on user input zip code). The model utilizes XGBoost, Logistic Regression, Fourier transformations, Bayesian optimization. This projects purpose is to showcase ability to work with limited data, technical proficiency, and innovation.

Features

    Feature Engineering:
        Fourier transforms to capture cyclical patterns.
        Temporal variables like hour, day of year, and timestamps.
        Extreme temperature thresholds for enhanced prediction sensitivity.

    Models:
        Primary XGBoost Model: Predicts general temperature trends.
        Extreme XGBoost Model: Focuses on rare high/low temperature events.
        Logistic Regression Meta Model: Dynamically blends predictions from the primary and extreme models.

    Optimization:
        Bayesian hyperparameter tuning for the XGBoost models.
        Fine-tuned thresholds for the logistic regression meta model.

    Visualization:
        Dynamic plots of future temperature predictions.

    Progress Tracking:
        A simple progress indicator is displayed in the console to indicate processing during time-intensive steps. 
    
    Output:
        Saves metrics and predictions to a structured CSV file for analysis.

How It Works

    Data Collection:
        Fetches historical weather data using the Open-Meteo API.
        Converts the data to a usable format with time zone adjustments.

    Feature Engineering:
        Adds cyclical features using Fourier transformations.
        Includes time-based features like hour and day_of_year.

    Model Training:
        Primary XGBoost: Learns general temperature patterns.
        Extreme XGBoost: Trained on the most extreme 5% of temperatures (top and bottom 2.5%).
        Logistic Regression: Weighs predictions dynamically based on extreme probabilities.

    Predictions:
        Generates predictions for the next week.
        Aggregates and evaluates results with metrics such as R², MAE, AUC-ROC, and Brier Score.

    Visualization and Export:
        Outputs a CSV file containing performance metrics and predictions.
        Visualizes temperature predictions for the next week using interactive plots.

Installation and Usage
Prerequisites

    Python 3.9 or later
    Install the required packages with:

    pip install -r weather_model_v1.1_requirements.txt

Running the Model

    Clone the repository:
      git clone https://github.com/Bryce-Ritt/Weather-ML-Model.git
      cd Weather-ML-Model
    
    Run the script:
      python "Weather Model (v1.1).py"
    
    Enter the ZIP code for your desired US location when prompted.

    Results:
        A CSV file (predictions_with_metrics.csv) will be saved in the project directory.
        A plot of the predictions will display in an interactive window.

File Output Details

The CSV file contains:

    Metrics: Performance metrics for each model (R², MAE, AUC-ROC, Brier Score).
    Predictions: Hourly predictions for the upcoming week.

Known Issues and Future Directions
Known Issues:

    The meta model struggles with dynamically weighting the extreme model accurately in all scenarios.
    The predictions may not fully capture rapid temperature shifts driven by unmodeled latent weather variables.
    Progress tracking could be improved to deliver more information about anticiated time to completion.

Planned Improvements:

    Explore alternative meta-model strategies, such as:
        Using residuals, lagged residuals, and interactions as features to train the meta model.
        Tweaking the Logistic Regression meta model to better capture temperature variations.
        Incorporating a more advanced meta model strategy is logistic regression cannot be used effectively.
        Improving the progress tracker to better inform users of how much longer the model will take to complete.
  
    Refine feature engineering:
        Consider adding residulas, lagged residuals, and interactions.
        Investigate the use of external features such as humidity or wind patterns.
        Test robustness across more diverse locations and seasons.

Contributions

    Core Development: Developed by Bryce Ritt.

Feedback and Contact

For questions, feedback, or suggestions, please open an issue in the repository.
