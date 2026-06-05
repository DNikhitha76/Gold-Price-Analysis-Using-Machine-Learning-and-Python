import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np

# Suppress warnings from seaborn distplot being deprecated
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")

# --- Backend Functions (Copied for self-contained Streamlit app) ---
# In a real-world scenario, you would import these from a separate file:
# from gold_analysis_backend import (
#     load_and_preprocess_data, get_descriptive_statistics,
#     get_correlation_matrix, plot_time_series, plot_distribution,
#     plot_correlation_heatmap, plot_scatter, train_and_predict_model,
#     predict_tomorrow_price
# )

# --- Data Loading and Preprocessing ---
@st.cache_data # Cache data to avoid reloading on every rerun
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
        st.error(f"Error: The file '{file_path}' was not found. Please ensure 'gld_price_data.csv' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred during data loading or preprocessing: {e}")
        return None

# --- Data Analysis Functions ---
@st.cache_data
def get_descriptive_statistics(df):
    """Returns descriptive statistics of the DataFrame."""
    return df.describe()

@st.cache_data
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
    Generates a distribution plot (histogram with KDE) for a given column using Plotly.
    """
    fig = px.histogram(df, x=column, nbins=30, marginal="box",
                       title=title, color_discrete_sequence=['green'])
    fig.update_layout(bargap=0.1)
    return fig

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
                        font=dict(color='white' if abs(value) > 0.5 else 'black', size=10)
                    )
                )
    fig.update_layout(annotations=annotations)
    return fig



# --- Machine Learning Model (RandomForestRegressor) ---
@st.cache_resource # Cache the model to avoid retraining on every rerun
def train_and_predict_model(df):
    """
    Trains a RandomForestRegressor model and makes predictions.
    Returns the trained model, test data, actual values, and predictions.
    """
    # Features (X) and Target (Y)
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
        feature_columns = ['SPX', 'USO', 'SLV', 'EUR/USD'] # Ensure this matches your training features
        input_df = pd.DataFrame([latest_data], columns=feature_columns)
        
        # Make prediction
        predicted_gld = model.predict(input_df)[0]
        return predicted_gld
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# --- Streamlit App Layout ---
def main():
    st.set_page_config(layout="wide", page_title="Gold Price Analysis")

    st.title("📈 Gold Price Analysis ")
    st.markdown("""
        This application performs an analysis of GLD (Gold ETF) prices,
        visualizes key financial metrics, and provides a prediction for tomorrow's GLD price
        using a RandomForestRegressor model.
    """)

    # Load data
    data = load_and_preprocess_data()

    if data is not None:
        st.sidebar.header("Navigation")
        analysis_options = [
            "Data Overview",
            "Time Series Analysis",
            "Distribution Analysis",
            "Correlation Analysis",
            "GLD Price Prediction"
        ]
        selected_analysis = st.sidebar.radio("Select Analysis Type:", analysis_options)

        # --- Data Overview ---
        if selected_analysis == "Data Overview":
            st.header("1. Data Overview")
            st.subheader("Raw Data (First 5 Rows)")
            st.dataframe(data.head())

            st.subheader("Raw Data (Last 5 Rows)")
            st.dataframe(data.tail())

            st.subheader("Dataset Shape")
            st.write(f"Number of rows: {data.shape[0]}, Number of columns: {data.shape[1]}")

            st.subheader("Data Information")
            st.text(data.info()) # st.text is used as st.info() is for messages

            st.subheader("Missing Values")
            st.dataframe(data.isnull().sum().to_frame(name='Missing Count'))

            st.subheader("Descriptive Statistics")
            st.dataframe(get_descriptive_statistics(data))

        # --- Time Series Analysis ---
        elif selected_analysis == "Time Series Analysis":
            st.header("2. Time Series Analysis")
            st.markdown("Visualize the historical trends of different financial instruments.")

            time_series_columns = data.columns.tolist()
            selected_ts_column = st.selectbox("Select a column to plot:", time_series_columns, index=time_series_columns.index('GLD'))
            
            st.plotly_chart(plot_time_series(data, selected_ts_column, f'{selected_ts_column} Price Over Time'), use_container_width=True)

        # --- Distribution Analysis ---
        elif selected_analysis == "Distribution Analysis":
            st.header("3. Distribution Analysis")
            st.markdown("Understand the distribution of financial instrument prices.")

            dist_columns = data.columns.tolist()
            selected_dist_column = st.selectbox("Select a column for distribution:", dist_columns, index=dist_columns.index('GLD'))

            st.plotly_chart(plot_distribution(data, selected_dist_column, f'Distribution of {selected_dist_column} Price'), use_container_width=True)

        # --- Correlation Analysis ---
        elif selected_analysis == "Correlation Analysis":
            st.header("4. Correlation Analysis")
            st.markdown("Explore the correlation between different financial instruments. A heatmap shows the strength and direction of linear relationships.")

            correlation_matrix = get_correlation_matrix(data)
            st.plotly_chart(plot_correlation_heatmap(correlation_matrix, 'Correlation Matrix of Financial Instruments'), use_container_width=True)

            st.subheader("GLD Correlations")
            if 'GLD' in correlation_matrix.columns:
                st.dataframe(correlation_matrix['GLD'].to_frame(name='Correlation with GLD'))
            else:
                st.write("GLD column not found in correlation matrix.")

        

        # --- GLD Price Prediction ---
        elif selected_analysis == "GLD Price Prediction":
            st.header("5. GLD Price Prediction")
            st.markdown("Predict tomorrow's GLD price based on today's values of other financial instruments using a trained RandomForestRegressor model.")

            st.subheader("Model Performance")
            model, X_test, Y_test, predictions, mse, r2 = train_and_predict_model(data)
            st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
            st.write(f"**R-squared (R²):** {r2:.2f}")
            st.info("A lower MSE indicates better prediction accuracy, and an R-squared closer to 1 indicates that the model explains more of the variance in the target variable.")

            st.subheader("Make a Prediction for Tomorrow")
            st.markdown("Enter the latest values for the following financial instruments to predict tomorrow's GLD price:")

            # Get the last available data point for pre-filling
            last_data_point = data.iloc[-1].to_dict()
            
            col1, col2 = st.columns(2)
            with col1:
                spx_val = st.number_input("SPX (S&P 500 Index)", value=float(last_data_point.get('SPX', 2725.52)), format="%.2f")
                uso_val = st.number_input("USO (United States Oil Fund)", value=float(last_data_point.get('USO', 13.56)), format="%.2f")
            with col2:
                slv_val = st.number_input("SLV (iShares Silver Trust)", value=float(last_data_point.get('SLV', 15.69)), format="%.2f")
                eur_usd_val = st.number_input("EUR/USD (Euro to US Dollar Exchange Rate)", value=float(last_data_point.get('EUR/USD', 1.21)), format="%.4f")

            if st.button("Predict GLD Price"):
                latest_input = {
                    'SPX': spx_val,
                    'USO': uso_val,
                    'SLV': slv_val,
                    'EUR/USD': eur_usd_val
                }
                
                with st.spinner('Predicting GLD price...'):
                    predicted_price = predict_tomorrow_price(model, latest_input)
                    if predicted_price is not None:
                        st.success(f"**Predicted GLD Price for Tomorrow:** ${predicted_price:.2f}")
                    else:
                        st.error("Could not make a prediction. Please check your input values.")

    st.sidebar.markdown("---")
    st.sidebar.info("Developed by Gold Price Analysis")

if __name__ == "__main__":
    main()
