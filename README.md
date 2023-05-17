# Data-Analysis-Tool
Dynamic Data Analysis Tool: An interactive Dash application for data analysis, including handling missing values, normalization, outlier detection, feature importance calculation, and various visualization tools such as correlation heatmaps and scatter plots.

# Dynamic Data Analysis Tool

This project presents an interactive Dash application for comprehensive data analysis. The tool includes several data preprocessing steps (like handling missing values, normalization), advanced analysis techniques (outlier detection, feature importance calculation), and various visualization tools (correlation heatmaps, scatter plots, and more).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The project requires Python 3.7 or later. You might want to use a virtual environment to manage the dependencies. You can create a virtual environment using the following command:

```bash
python -m venv env
```

Then, activate the virtual environment.

On Windows:

```
.\env\Scripts\activate
```

On Linux or MacOS:

```
source env/bin/activate
```

# Installing
Once your virtual environment is activated, install the project dependencies with:

```
pip install -r requirements.txt
```

# Running the Application
To start the Dash application, run:

```
python main.py
```

or 

```
python3 main.py
```

This will start a local server, and you can interact with the application by navigating to the provided URL (usually http://127.0.0.1:8050/).

# Customizing the CSV File

The CSV file name is set in the main function in main.py file. To change the input CSV file, you can replace 'data.csv' in the following line:

```
analyzer = DataAnalyzer('data.csv')
```

with the path to your CSV file, like so:

```
analyzer = DataAnalyzer('path_to_your_file.csv')
```

Be sure to use the relative path from main.py to your CSV file.

# License

This project is licensed under the MIT License.
