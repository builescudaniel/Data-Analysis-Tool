import pandas as pd
import seaborn as sns
import plotly.express as px
import dash
import dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from scipy.stats import zscore
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np


class DataAnalyzer:
    """DataAnalyzer class for performing data analysis."""

    def __init__(self, file_name):
        """Constructor method. Loads and processes data."""

        # Load the data from the file
        self.data = self.load_data(file_name)

        # Handle missing values in the data
        self.missing_values, self.data = self.handle_missing_values()

        # Normalize the data
        self.data = self.normalize_data()

        # Generate summary statistics of the data
        self.summary = self.data.describe(include='all')

        # Find the top 5 records
        self.top_records = self.data.nlargest(5, self.data.columns[0])

        # Find the bottom 5 records
        self.bottom_records = self.data.nsmallest(5, self.data.columns[0])

        # Detect outliers in the data
        self.outliers = self.detect_outliers()

        # Calculate feature importance
        self.importance = self.feature_importance()

        # Create pairwise scatter plots
        self.scatter_plots = self.pairwise_scatter_plots()

    def load_data(self, file):
        """Loads data from a CSV file."""
        return pd.read_csv(file)

    def handle_missing_values(self):
        """Handles missing values in the dataset."""
        missing_values = self.data.isnull().sum()
        data = self.data.fillna(self.data.median())
        return missing_values, data

    def normalize_data(self):
        """Normalizes numeric data in the dataset."""
        for col in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                self.data[col] = (self.data[col] - self.data[col].min()) / \
                    (self.data[col].max() - self.data[col].min())
        return self.data

    def detect_outliers(self):
        """Detects outliers using Z-score method."""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        outliers = self.data[numeric_cols].apply(zscore).abs() > 3
        return outliers

    def feature_importance(self):
        """Calculates feature importance using a Random Forest Regressor."""
        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        importance = pd.Series(model.feature_importances_,
                               index=X.columns).sort_values(ascending=False)
        return importance

    def pairwise_scatter_plots(self):
        """Creates pairwise scatter plots."""
        fig = px.scatter_matrix(self.data, dimensions=self.data.columns,
                                title='Pairwise Scatter Plots')
        return fig


class DashApp:
    """DashApp class for creating and running the Dash application."""

    def __init__(self, analyzer):
        """Constructor method. Creates a Dash application."""
        self.analyzer = analyzer
        self.app = dash.Dash(__name__, external_stylesheets=[
                             dbc.themes.BOOTSTRAP])
        self.app.layout = self.create_layout()

    def create_layout(self):
        """Creates the layout for the Dash application."""
        return html.Div([
            html.H1('Dynamic Data Analysis Tool'),
            html.H2('Data'),
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i}
                         for i in self.analyzer.data.columns],
                data=self.analyzer.data.to_dict('records'),
                page_size=10,
                style_table={'overflowX': 'auto'}
            ),
            html.H2('Missing Values Report'),
            dash_table.DataTable(
                id='missing_values',
                columns=[{"name": i, "id": i}
                         for i in self.analyzer.missing_values.index],
                data=self.analyzer.missing_values.reset_index().rename(
                    columns={'index': 'Column', 0: 'Missing Values'}).to_dict('records'),
                style_table={'overflowX': 'auto'}
            ),
            html.H2('Summary Statistics'),
            dash_table.DataTable(
                id='summary',
                columns=[{"name": i, "id": i}
                         for i in self.analyzer.summary.columns],
                data=self.analyzer.summary.reset_index().to_dict('records'),
                style_table={'overflowX': 'auto'}
            ),
            html.H2('Top 5 Records'),
            dash_table.DataTable(
                id='top_records',
                columns=[{"name": i, "id": i}
                         for i in self.analyzer.top_records.columns],
                data=self.analyzer.top_records.to_dict('records'),
                style_table={'overflowX': 'auto'}
            ),
            html.H2('Bottom 5 Records'),
            dash_table.DataTable(
                id='bottom_records',
                columns=[{"name": i, "id": i}
                         for i in self.analyzer.bottom_records.columns],
                data=self.analyzer.bottom_records.to_dict('records'),
                style_table={'overflowX': 'auto'}
            ),
            html.H2('Correlation Heatmap'),
            dcc.Graph(
                id='heatmap',
                figure=px.imshow(self.analyzer.data.corr(),
                                 title='Correlation Heatmap')
            ),
            html.H2('Outliers'),
            dash_table.DataTable(
                id='outliers',
                columns=[{"name": i, "id": i}
                         for i in self.analyzer.outliers.columns],
                data=self.analyzer.outliers.to_dict('records'),
                style_table={'overflowX': 'auto'}
            ),
            html.H2('Feature Importance'),
            dcc.Graph(
                id='importance',
                figure=px.bar(self.analyzer.importance, x=self.analyzer.importance.index,
                              y=self.analyzer.importance.values, title='Feature Importance')
            ),
            html.H2('Pairwise Scatter Plots'),
            dcc.Graph(
                id='scatter_plots',
                figure=self.analyzer.scatter_plots
            )
        ])

    def run(self):
        """Runs the Dash application."""
        self.app.run_server(debug=True)


def main():
    """Main function to run the data analysis."""
    # Initialize a DataAnalyzer object
    analyzer = DataAnalyzer('data.csv')

    # Initialize a DashApp object with the DataAnalyzer object as input
    app = DashApp(analyzer)

    # Run the Dash app
    app.run()


if __name__ == "__main__":
    main()
