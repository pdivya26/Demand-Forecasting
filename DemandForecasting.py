import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('C:\\Users\\divya\\Documents\\Academics\\Datasets\\RetailStoreInventory.csv')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    # Fill missing values
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Base temporal features
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['DaysSinceStart'] = (df['Date'] - df['Date'].min()).dt.days

    # Binary flag from 'Holiday/Promotion' (if available)
    if 'Holiday/Promotion' in df.columns:
        df['IsHoliday'] = df['Holiday/Promotion'].apply(lambda x: 1 if 'holiday' in str(x).lower() or 'promo' in str(x).lower() else 0)

    # Log transformation
    for col in ['Units Sold', 'Discount']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: np.log1p(x))

    # Outlier detection and removal for Units Sold (after log1p)
    Q1 = df['Units Sold'].quantile(0.25)
    Q3 = df['Units Sold'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['Units Sold'] >= lower_bound) & (df['Units Sold'] <= upper_bound)]

    # Feature engineering: Lag and Rolling mean
    df = df.sort_values(by=['Product ID', 'Date'])
    df['Lag_1'] = df.groupby('Product ID')['Units Sold'].shift(1)
    df['Rolling_3'] = df.groupby('Product ID')['Units Sold'].shift(1).rolling(window=3).mean()

    # Fill NaNs created by lag/rolling
    df[['Lag_1', 'Rolling_3']] = df[['Lag_1', 'Rolling_3']].fillna(method='bfill').fillna(0)

    # One-hot encoding for selected categoricals
    cat_features = ['Region', 'Weather Condition', 'Seasonality']
    cat_features = [col for col in cat_features if col in df.columns]
    df = pd.get_dummies(df, columns=cat_features, drop_first=True)

    return df

df = load_data()

# Streamlit UI
st.markdown(f"<h1 style='text-align: center;'>Demand Forecasting</h3>", unsafe_allow_html=True)
available_categories = df['Category'].unique()
selected_category = st.selectbox("Select a Product Category:", available_categories)

df_selected = df[df['Category'] == selected_category]

if df_selected.empty:
    st.warning(f"No data found for category: {selected_category}")
else:
    df_selected['Units Sold'] = np.expm1(df_selected['Units Sold'])

    # Monthly Bar Chart
    st.markdown(f"<h3 style='text-align: center;'>Monthly Total Units Sold - {selected_category}</h3>", unsafe_allow_html=True)

    df_selected['MonthName'] = df_selected['Date'].dt.strftime('%b')
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_demand = df_selected.groupby('MonthName')['Units Sold'].sum().reindex(month_order)
    fig1, ax1 = plt.subplots(figsize=(20, 10))
    bars = monthly_demand.plot(kind='bar', color='teal', edgecolor='teal', ax=ax1)
    ax1.set_xlabel("Month", fontsize=12)
    ax1.set_ylabel("Units Sold", fontsize=12)
    ax1.set_title(f"Monthly Total Units Sold - {selected_category}", fontsize=14)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
    for bar in bars.patches:
        height = bar.get_height()
        ax1.annotate(f'{int(height/1000)}k',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig1)

    # Yearly Bar Chart
    st.markdown(f"<h3 style='text-align: center;'>Yearly Total Units Sold - {selected_category}</h3>", unsafe_allow_html=True)
    
    yearly_demand = df_selected.groupby(df_selected['Date'].dt.year)['Units Sold'].sum()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bars = yearly_demand.plot(kind='bar', color='teal', edgecolor='teal', ax=ax2)
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Units Sold", fontsize=12)
    ax2.set_title(f"Yearly Total Units Sold - {selected_category}", fontsize=14)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
    for bar in bars.patches:
        height = bar.get_height()
        ax2.annotate(f'{int(height/1000)}k',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2)

    # Monthly Trend Line Plot
    st.markdown(f"<h3 style='text-align: center;'>Monthly Sales Trend - {selected_category}</h3>", unsafe_allow_html=True)

    df_selected['MonthPeriod'] = df_selected['Date'].dt.to_period('M')
    trend = df_selected.groupby('MonthPeriod')['Units Sold'].sum()
    trend.index = trend.index.to_timestamp()
    trend.index = trend.index.strftime('%b %Y')
    fig3, ax3 = plt.subplots(figsize=(20, 10))
    ax3.plot(trend.index, trend.values, marker='o', linestyle='-', color='teal', linewidth=2)
    ax3.set_xlabel("Month", fontsize=12)
    ax3.set_ylabel("Units Sold", fontsize=12)
    ax3.set_title(f"Monthly Sales Trend - {selected_category}", fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    st.pyplot(fig3)

    # Modeling
    df_cat = df[df['Category'] == selected_category]
    # Create a copy for lag and rolling calculations before one-hot encoding
    df_cat_raw = df_cat.copy()
    df_cat = pd.get_dummies(df_cat, columns=['Product ID'], drop_first=True)

    if df_cat.shape[0] < 30:
        st.error(f"Not enough data for category '{selected_category}' to train model.")
    else:
        X = df_cat.drop(columns=['Units Sold', 'Category', 'Date', 'Store ID'])
        y = df_cat['Units Sold']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Get top N important features using Random Forest
        rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_temp.fit(X, y)

        # Get feature importances
        importances = pd.Series(rf_temp.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)

        # Select top N features
        top_n = 10
        important_features = importances.head(top_n).index.tolist()
        print(f"\nTop {top_n} features selected for modeling:\n", important_features)

        # Reduce feature set
        X = X[important_features]

        # Redefine train/test split using only important features
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        results = {}
        
        st.markdown(f"<h2 style='text-align: center;'>Actual vs Predicted Units Sold</h2>", unsafe_allow_html=True)

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

            test_dates = df_cat.loc[X_test.index, 'Date']
            results_df = pd.DataFrame({
                'Date': test_dates.values,
                'Actual': np.expm1(y_test.values),
                'Predicted': np.expm1(preds)
            })
            results_df['Month'] = pd.to_datetime(results_df['Date']).dt.to_period('M')
            monthly_results = results_df.groupby('Month')[['Actual', 'Predicted']].sum()
            monthly_results.index = monthly_results.index.to_timestamp()
            monthly_results.index = monthly_results.index.strftime('%b %Y')

            st.subheader(f"{name}")
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(monthly_results.index, monthly_results['Actual'], marker='o', label='Actual', color='green', linewidth=2)
            ax.plot(monthly_results.index, monthly_results['Predicted'], marker='s', linestyle='--', label='Predicted', color='red', linewidth=2)
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Units Sold', fontsize=12)
            ax.set_title(f'Actual vs Predicted - {name}', fontsize=14)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
            plt.xticks(rotation=45, fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)

        # Future Sales Prediction for the Next Month (Simplified)
        st.markdown(f"<h2 style='text-align: center;'>Future Sales Prediction for Next Month - {selected_category}</h2>", unsafe_allow_html=True)

        # Determine the last date in the dataset
        last_date = df_cat_raw['Date'].max()
        future_date = last_date + pd.offsets.MonthEnd(1) + pd.offsets.MonthBegin(1)

        # Prepare future data using df_cat_raw (before one-hot encoding)
        # Select the most recent data for each Product ID
        future_df = df_cat_raw.sort_values('Date').groupby('Product ID').tail(1).copy()
        future_df['Date'] = future_date
        future_df['Month'] = future_date.month
        future_df['Year'] = future_date.year
        future_df['DayOfWeek'] = future_date.dayofweek
        future_df['IsWeekend'] = 1 if future_date.dayofweek in [5, 6] else 0
        future_df['DaysSinceStart'] = (future_date - df['Date'].min()).days
        future_df['IsHoliday'] = df_cat_raw['IsHoliday'].mode()[0]  # Use most common holiday status

        # Use existing Lag_1 and Rolling_3 from the most recent data
        # If not available, compute from historical data
        if 'Lag_1' not in future_df.columns or 'Rolling_3' not in future_df.columns:
            lag_1 = df_cat_raw.sort_values('Date').groupby('Product ID')['Units Sold'].last()
            rolling_3 = df_cat_raw.sort_values('Date').groupby('Product ID')['Units Sold'].tail(3).groupby('Product ID').mean()
            future_df['Lag_1'] = future_df['Product ID'].map(lag_1)
            future_df['Rolling_3'] = future_df['Product ID'].map(rolling_3)

        # Fill any NaNs in Lag_1 and Rolling_3
        future_df[['Lag_1', 'Rolling_3']] = future_df[['Lag_1', 'Rolling_3']].fillna(df_cat_raw[['Lag_1', 'Rolling_3']].median())

        # One-hot encode categorical features for future data
        future_df = pd.get_dummies(future_df, columns=['Product ID'], drop_first=True)

        # Align future_df columns with X_train
        for col in X_train.columns:
            if col not in future_df.columns:
                future_df[col] = 0
        future_df = future_df[X_train.columns]

        # Predict future sales
        future_preds = {}
        for name, model in models.items():
            preds = model.predict(future_df[important_features])
            future_preds[name] = np.expm1(preds).sum()

        # Create a DataFrame for future predictions
        future_results = pd.DataFrame({
            'Model': list(future_preds.keys()),
            'Predicted Units Sold': list(future_preds.values())
        })
        future_results['Predicted Units Sold'] = future_results['Predicted Units Sold'].round()

        # Visualize future predictions
        st.markdown(f"<h3 style='text-align: center;'>Predicted Sales for {future_date.strftime('%B %Y')}", unsafe_allow_html=True)
        fig_future, ax_future = plt.subplots(figsize=(10, 6))
        bars = ax_future.bar(future_results['Model'], future_results['Predicted Units Sold'], color='teal', edgecolor='teal')
        ax_future.set_xlabel("Model", fontsize=12)
        ax_future.set_ylabel("Predicted Units Sold", fontsize=12)
        ax_future.set_title(f"Predicted Sales for {future_date.strftime('%B %Y')} - {selected_category}", fontsize=14)
        ax_future.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x}s'))
        for bar in bars:
            height = bar.get_height()
            ax_future.annotate(f'{height}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_future)

        st.markdown("<h3 style='text-align: center;'>Future Sales Prediction Metrics</h3>", unsafe_allow_html=True)
        with st.container():
            st.dataframe(
                future_results.style.format({
                "Predicted Units Sold": "{:.0f}"
                }),
                use_container_width=True
            )

        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>Model Performance Metrics</h3>", unsafe_allow_html=True)

        perf_df = pd.DataFrame(results).T

        with st.container():
            st.dataframe(
                perf_df.style.format({
                    "MAE": "{:.2f}",
                    "RMSE": "{:.2f}",
                    "R2": "{:.2%}"
                }),
                use_container_width=True
            )