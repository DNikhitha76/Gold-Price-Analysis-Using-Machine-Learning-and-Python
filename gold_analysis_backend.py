import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import plotly.express as px
import plotly.graph_objects as go

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(file_path='gld_price_data.csv'):
    """
    Loads the gold price data from a CSV file and preprocesses it.
    - Converts 'Date' column to datetime objects.
    - Handles any missing values (though the notebook indicated none).
    """
    try:
        gold_data = pd.read_csv(file_path)
        # Convert 'Date' to datetime objects for time series analysis
        gold_data['Date'] = pd.to_datetime(gold_data['Date'])
        # Set 'Date' as index for time series plotting
        gold_data.set_index('Date', inplace=True)
        return gold_data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred during data loading or preprocessing: {e}")
        return None

# --- Data Analysis Functions ---
def get_descriptive_statistics(df):
    """Returns descriptive statistics of the DataFrame."""
    return df.describe()

def get_correlation_matrix(df):
    """
    Calculates and returns the correlation matrix for numerical columns.
    Excludes the 'Date' column if it's still present as a column (should be index).
    """
    # Select only numeric columns for correlation calculation
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr(method='pearson')

# --- Data Visualization Functions ---
def plot_time_series(df, column='GLD', title='GLD Price Over Time'):
    """
    Generates an interactive time series plot for a given column using Plotly.
    """
    fig = px.line(df, x=df.index, y=column, title=title)
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.update_layout(hovermode="x unified")
    return fig

def plot_distribution(df, column='GLD', title='Distribution of GLD Price'):
    """
    Generates a distribution plot (histogram with KDE) for a given column using Seaborn.
    Returns a matplotlib figure.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, color='green', bins=30)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency / Density')
    plt.grid(True, linestyle='--', alpha=0.6)
    return plt.gcf() # Get current figure to return

def plot_correlation_heatmap(correlation_matrix, title='Correlation Matrix'):
    """
    Creates an optimized correlation heatmap using Plotly.
    """
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    # Apply mask to the correlation matrix
    masked_corr = correlation_matrix.mask(mask)

    fig = go.Figure(data=go.Heatmap(
        z=masked_corr.values,
        x=masked_corr.columns.tolist(),
        y=masked_corr.index.tolist(),
        colorscale='Blues',
        zmin=-1, zmax=1,
        hoverongaps = False
    ))

    fig.update_layout(
        title_text=title,
        xaxis_nticks=len(masked_corr.columns),
        yaxis_nticks=len(masked_corr.index),
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        yaxis_autorange='reversed' # To display y-axis in correct order
    )

    # Add text annotations for correlation values
    annotations = []
    for i, row in enumerate(masked_corr.values):
        for j, value in enumerate(row):
            if not np.isnan(value): # Only add annotation if not masked
                annotations.append(
                    dict(
                        x=masked_corr.columns[j],
                        y=masked_corr.index[i],
                        text=f'{value:.2f}',
                        showarrow=False,
                        font=dict(color='white' if abs(value) > 0.5 else 'black') # White text for dark blues, black for light
                    )
                )
    fig.update_layout(annotations=annotations)
    return fig



# --- Machine Learning Model (RandomForestRegressor) ---
def train_and_predict_model(df):
    """
    Trains a RandomForestRegressor model and makes predictions.
    Returns the trained model, test data, actual values, and predictions.
    """
    # Features (X) and Target (Y)
    # Drop 'Date' (already set as index) and 'GLD' for features
    X = df.drop(columns=['GLD'])
    Y = df['GLD']

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Initialize and train the RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)

    # Make predictions on the test data
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = metrics.mean_squared_error(Y_test, predictions)
    r2_score = metrics.r2_score(Y_test, predictions)

    return model, X_test, Y_test, predictions, mse, r2_score

def predict_tomorrow_price(model, latest_data):
    """
    Predicts tomorrow's GLD price based on the trained model and latest input data.
    latest_data should be a dictionary with keys matching model's feature columns.
    """
    try:
        # Convert the dictionary to a DataFrame, ensuring column order matches training data
        # The order of columns in X_test is important for prediction
        feature_columns = ['SPX', 'USO', 'SLV', 'EUR/USD'] # Ensure this matches your training features
        input_df = pd.DataFrame([latest_data], columns=feature_columns)
        
        # Make prediction
        predicted_gld = model.predict(input_df)[0]
        return predicted_gld
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

if __name__ == '__main__':
    # Example usage for testing backend functions
    data = load_and_preprocess_data()
    if data is not None:
        print("Data loaded successfully:")
        print(data.head())

        print("\nDescriptive Statistics:")
        print(get_descriptive_statistics(data))

        print("\nCorrelation Matrix:")
        corr_matrix = get_correlation_matrix(data)
        print(corr_matrix)

        # Example of plotting (these will open matplotlib windows if run directly)
        # plot_time_series(data, 'GLD')
        # plot_distribution(data, 'GLD')
        # plot_correlation_heatmap(corr_matrix)
        # plot_scatter(data, 'SPX', 'GLD')

        # Train model and get predictions
        model, X_test, Y_test, predictions, mse, r2 = train_and_predict_model(data)
        print(f"\nModel Mean Squared Error: {mse:.2f}")
        print(f"Model R-squared: {r2:.2f}")

        # Example prediction for tomorrow
        # Use the last row of the dataset as 'latest_data' for demonstration
        # In a real app, this would come from user input or a live data source
        latest_features = data.drop(columns=['GLD']).iloc[-1].to_dict()
        print(f"\nLatest data for prediction: {latest_features}")
        predicted_gld_price = predict_tomorrow_price(model, latest_features)
        if predicted_gld_price is not None:
            print(f"Predicted GLD price for tomorrow: {predicted_gld_price:.2f}")
