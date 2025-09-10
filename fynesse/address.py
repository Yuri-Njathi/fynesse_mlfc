"""
Address module for the fynesse framework.

This module handles question addressing functionality including:
- Statistical analysis
- Predictive modeling
- Data visualization for decision-making
- Dashboard creation
"""

from typing import Any, Union
import pandas as pd
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Here are some of the imports we might expect
# import sklearn.model_selection  as ms
# import sklearn.linear_model as lm
# import sklearn.svm as svm
# import sklearn.naive_bayes as naive_bayes
# import sklearn.tree as tree

# import GPy
# import torch
# import tensorflow as tf

# Or if it's a statistical analysis
# import scipy.stats


def analyze_data(data: Union[pd.DataFrame, Any]) -> dict[str, Any]:
    """
    Address a particular question that arises from the data.

    IMPLEMENTATION GUIDE FOR STUDENTS:
    ==================================

    1. REPLACE THIS FUNCTION WITH YOUR ANALYSIS CODE:
       - Perform statistical analysis on the data
       - Create visualizations to explore patterns
       - Build models to answer specific questions
       - Generate insights and recommendations

    2. ADD ERROR HANDLING:
       - Check if input data is valid and sufficient
       - Handle analysis failures gracefully
       - Validate analysis results

    3. ADD BASIC LOGGING:
       - Log analysis steps and progress
       - Log key findings and insights
       - Log any issues encountered

    4. EXAMPLE IMPLEMENTATION:
       if data is None or len(data) == 0:
           print("Error: No data available for analysis")
           return {}

       print("Starting data analysis...")
       # Your analysis code here
       results = {"sample_size": len(data), "analysis_complete": True}
       return results
    """
    logger.info("Starting data analysis")

    # Validate input data
    if data is None:
        logger.error("No data provided for analysis")
        print("Error: No data available for analysis")
        return {"error": "No data provided"}

    if len(data) == 0:
        logger.error("Empty dataset provided for analysis")
        print("Error: Empty dataset provided for analysis")
        return {"error": "Empty dataset"}

    logger.info(f"Analyzing data with {len(data)} rows, {len(data.columns)} columns")

    try:
        # STUDENT IMPLEMENTATION: Add your analysis code here

        # Example: Basic data summary
        results = {
            "sample_size": len(data),
            "columns": list(data.columns),
            "data_types": data.dtypes.to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "analysis_complete": True,
        }

        # Example: Basic statistics (students should customize this)
        numeric_columns = data.select_dtypes(include=["number"]).columns
        if len(numeric_columns) > 0:
            results["numeric_summary"] = data[numeric_columns].describe().to_dict()

        logger.info("Data analysis completed successfully")
        print(f"Analysis completed. Sample size: {len(data)}")

        return results

    except Exception as e:
        logger.error(f"Error during data analysis: {e}")
        print(f"Error analyzing data: {e}")
        return {"error": str(e)}

def cross_entropy(y_true, y_pred):
    #TODO
    N = len(y_true)
    y_pred = np.array(y_pred)+1e-24
    y_true = np.array(y_true)
    # Calculate the cross-entropy loss
    return -(1.0/N) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    ##raise NotImplementedError("Cross entropy not implemented yet.")

def evaluate_prediction_system(df, your_function, max_samples=1000):
    # Randomly sample up to 1000, if full evaluation taking too long
    np.random.seed(42)
    coords = [(date, camera, species) for date in df.index for (camera, species) in df.columns]
    if len(coords) > max_samples:
        coords = np.random.choice(len(coords), size=max_samples, replace=False)
        coords = [coords[i] if isinstance(coords[i], tuple) else
                  [(date, camera, species) for date in df.index for (camera, species) in df.columns][coords[i]]
                  for i in range(len(coords))]
    else:
        coords = coords

    y_true = []
    y_pred = []

    for date, camera, species in coords:
        # print(date, camera, species)
        value = df.loc[date, (camera, species)]
        # print(value)
        y_true.append(value)
        prob = bayes_sighting_probability(df, camera, species, date)
        y_pred.append(prob)
        #break
    #print(y_pred)
    #print(np.mean(y_true))
    return cross_entropy(y_true, y_pred)
def bayes_sighting_probability(df, camera, species, date,debug=False) -> float:
    """
    Removes a specific observation and estimates the probability of sighting
    a given species at a given camera on a specific date.

    Parameters:
        df (pd.DataFrame): DataFrame with MultiIndex columns (camera, species) and datetime.date index.
        camera (str): Camera ID (e.g. 'C001').
        species (str): Species name (e.g. 'IMPALA').
        date (str or datetime.date or pd.Timestamp): Date of the observation.

    Returns:
        float: Estimated sighting probability - TODO.
    """
    # if isinstance(date, str) or isinstance(date, pd.Timestamp):
    #     date = pd.to_datetime(date).date()

    df_blind = df.copy()
    
    df_blind.loc[date, (camera, species)] = None #df blind table of interest

    probability_of_a_sighting = df_blind.mean().mean()
    
    prob_of_sighting_given_camera = df_blind.loc[:,(camera,slice(None))].mean().mean()
    #print(df_blind.loc[date,:].head())
    
    prob_of_sighting_given_species = df_blind.loc[:,(slice(None),species)].mean().mean()
    
    prob_of_sighting_given_date = df_blind.loc[date,:].mean()
    
    #TODO
    if debug:
      print("probability of a sighting",probability_of_a_sighting)
      print("prob of sighting given date",prob_of_sighting_given_date)
      print("prob of sighting given species",prob_of_sighting_given_species)
      print("prob of sighting given camera",prob_of_sighting_given_camera)
    naive_bayes_probability = prob_of_sighting_given_camera*prob_of_sighting_given_species*prob_of_sighting_given_date/probability_of_a_sighting**2
    #print("naive bayes probability",naive_bayes_probability)
    return naive_bayes_probability
