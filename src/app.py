import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os

from pathlib import Path
import joblib

# Get the directory where the script is located
script_dir = Path(__file__).parent

# Define paths relative to the script directory
data_dir = script_dir.parent / "data"
artifacts_dir = script_dir.parent / "artifacts"

# Create directories if they don't exist
data_dir.mkdir(parents=True, exist_ok=True)
artifacts_dir.mkdir(parents=True, exist_ok=True)

# Define file paths
test_data = data_dir / 'test.csv'
m1 = artifacts_dir / 'model_1.pkl'
m2 = artifacts_dir / 'model_2.pkl'



# Initialize the Dash app
app = dash.Dash(__name__, title="Student Grade Prediction Dashboard")
server = app.server

# Define colors
colors = {
    'background': '#f9f9f9',
    'text': '#333333',
    'primary': '#007bff',
    'secondary': '#6c757d',
    'success': '#28a745',
    'info': '#17a2b8',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'light': '#f8f9fa',
    'dark': '#343a40',
}

# Function to load models
def load_models():
    try:
        model_1 = joblib.load(m1)
        model_2 = joblib.load(m2)
        return (model_1, model_2)
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

# Function to load test data
def load_test_data():
    try:
        data = pd.read_csv(test_data)
        return data
    except Exception as e:
        print(f"Error loading test data: {e}")
        # Create sample data if file doesn't exist
        return pd.DataFrame({
            'StudyTimeWeekly': [19.83, 15.40, 4.21, 10.02, 4.67],
            'Absences': [7, 0, 26, 14, 17],
            'Tutoring': [1, 0, 0, 0, 1],
            'ParentalSupport': [2, 1, 2, 3, 3],
            'GradeClass': [2.0, 1.0, 4.0, 3.0, 4.0],
            'EngagementIndex': [1, 0, 0, 1, 0],
            'AttendanceRate': [96.11, 100.0, 85.55, 92.22, 90.55]
        })

# Load models and data
model_1, model_2 = load_models()


print()
test_data = load_test_data()
X_test = test_data.drop(['GradeClass'], axis=1)
y_pred = model_1.predict(X_test)
print(y_pred)


# Function to preprocess data for prediction
def preprocess_data(input_data):
    # Create a DataFrame from input values
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()
    
    # Extract features (exclude target variable)
    if 'GradeClass' in df.columns:
        X = df.drop('GradeClass', axis=1)
    else:
        X = df
    
    # Standardize features if needed for deep learning model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X, X_scaled

# Function to make predictions
def make_predictions(input_data):
    try:
        X, X_scaled = preprocess_data(input_data)
        
        # Random Forest prediction
        rf_pred = model_1.predict(X)[0] if model_1 is not None else 2.0
        
        # Deep Learning prediction
        dl_pred = model_2.predict(X_scaled)[0] if model_2 is not None else 2.0
        
        # If the prediction is a numpy array with multiple elements, take the first one
        if hasattr(dl_pred, "__len__") and len(dl_pred) > 0:
            dl_pred = dl_pred[0]
            
        # Round deep learning prediction to nearest grade point
        dl_pred = round(float(dl_pred) * 2) / 2
        
        # Ensure predictions are within valid grade range
        rf_pred = max(0.0, min(4.0, rf_pred))
        dl_pred = max(0.0, min(4.0, dl_pred))
        
        return rf_pred, dl_pred
    except Exception as e:
        print(f"Error making predictions: {e}")
        return 2.0, 2.0

# Function to get letter grade from numeric grade
def get_letter_grade(grade):
    if grade >= 3.5:
        return 'A'
    elif grade >= 2.5:
        return 'B'
    elif grade >= 1.5:
        return 'C'
    elif grade >= 0.5:
        return 'D'
    else:
        return 'F'

# App layout
app.layout = html.Div(style={'backgroundColor': colors['background'], 'padding': '20px'}, children=[
    html.Div([
        html.H1("Student Grade Prediction Dashboard", 
                style={'textAlign': 'center', 'color': colors['dark'], 'marginBottom': '30px'}),
        
        html.Div([
            html.Div([
                html.H3("Input Student Data", style={'color': colors['primary']}),
                html.Div([
                    html.Label("Study Time Weekly (hours)"),
                    dcc.Input(id='study-time', type='number', value=10.0, min=0, max=40, step=0.5,
                             style={'width': '100%', 'marginBottom': '10px'})
                ]),
                html.Div([
                    html.Label("Absences (count)"),
                    dcc.Input(id='absences', type='number', value=5, min=0, max=100, step=1,
                             style={'width': '100%', 'marginBottom': '10px'})
                ]),
                html.Div([
                    html.Label("Tutoring"),
                    dcc.Dropdown(id='tutoring', options=[
                        {'label': 'Yes', 'value': 1},
                        {'label': 'No', 'value': 0}
                    ], value=0, style={'width': '100%', 'marginBottom': '10px'})
                ]),
                html.Div([
                    html.Label("Parental Support Level"),
                    html.Div(style={'marginBottom': '20px'}, children=[
                        dcc.Slider(
                            id='parental-support',
                            min=1,
                            max=3,
                            value=2,
                            marks={1: '1', 2: '2', 3: '3', 4: '4'},
                            step=1,
                            included=True
                        )
                    ])
                ]),
                html.Div([
                    html.Label("Engagement Index"),
                    dcc.Dropdown(id='engagement', options=[
                        {'label': 'High', 'value': 1},
                        {'label': 'Low', 'value': 0}
                    ], value=1, style={'width': '100%', 'marginBottom': '10px'})
                ]),
                html.Div([
                    html.Label("Attendance Rate (%)"),
                    dcc.Input(id='attendance', type='number', value=95.0, min=0, max=100, step=0.1,
                             style={'width': '100%', 'marginBottom': '20px'})
                ]),
                html.Button('Predict Grade', id='predict-button', n_clicks=0,
                           style={'backgroundColor': colors['primary'], 'color': 'white', 
                                  'padding': '10px', 'borderRadius': '5px', 'width': '100%'})
            ], style={'width': '30%', 'padding': '20px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)', 
                      'borderRadius': '10px', 'backgroundColor': colors['light']}),
            
            html.Div([
                html.H3("Prediction Results", style={'color': colors['primary']}),
                html.Div(id='prediction-output', style={'marginBottom': '20px'}),
                
                dcc.Graph(id='prediction-gauge')
            ], style={'width': '65%', 'padding': '20px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)', 
                      'borderRadius': '10px', 'backgroundColor': colors['light']})
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '30px'}),
        
        html.Div([
            html.H3("Model Comparison on Test Data", style={'color': colors['primary']}),
            dcc.Graph(id='model-comparison-graph'),
            
            html.Div([
                html.H4("Test Data Sample", style={'color': colors['secondary']}),
                dash_table.DataTable(
                    id='test-data-table',
                    columns=[{'name': col, 'id': col} for col in test_data.columns],
                    data=test_data.head(10).to_dict('records'),
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'center', 'padding': '5px'},
                    style_header={
                        'backgroundColor': colors['primary'],
                        'color': 'white',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': colors['light']
                        }
                    ]
                )
            ], style={'marginTop': '20px'})
        ], style={'padding': '20px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)', 
                  'borderRadius': '10px', 'backgroundColor': colors['light']}),
        
        html.Div([
            html.H3("Feature Importance", style={'color': colors['primary']}),
            dcc.Graph(id='feature-importance')
        ], style={'marginTop': '30px', 'padding': '20px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)', 
                  'borderRadius': '10px', 'backgroundColor': colors['light']})
    ])
])

# Callback for prediction
@app.callback(
    [Output('prediction-output', 'children'),
     Output('prediction-gauge', 'figure')],
    [Input('predict-button', 'n_clicks')],
    [State('study-time', 'value'),
     State('absences', 'value'),
     State('tutoring', 'value'),
     State('parental-support', 'value'),
     State('engagement', 'value'),
     State('attendance', 'value')]
)
def update_prediction(n_clicks, study_time, absences, tutoring, parental_support, engagement, attendance):
    # Create input data dictionary
    input_data = {
        'StudyTimeWeekly': study_time,
        'Absences': absences,
        'Tutoring': tutoring,
        'ParentalSupport': parental_support,
        'EngagementIndex': engagement,
        'AttendanceRate': attendance
    }
    
    # Make predictions
    rf_pred, dl_pred = make_predictions(input_data)
    
    # Create prediction output
    rf_letter = get_letter_grade(rf_pred)
    dl_letter = get_letter_grade(dl_pred)
    
    prediction_output = html.Div([
        html.Div([
            html.H4("Random Forest Model", style={'color': colors['info']}),
            html.H2(f"{rf_pred:.1f} ({rf_letter})", style={'color': colors['dark'], 'textAlign': 'center'})
        ], style={'width': '45%', 'padding': '15px', 'border': f'2px solid {colors["info"]}', 'borderRadius': '5px'}),
        
        html.Div([
            html.H4("Deep Learning Model", style={'color': colors['success']}),
            html.H2(f"{dl_pred:.1f} ({dl_letter})", style={'color': colors['dark'], 'textAlign': 'center'})
        ], style={'width': '45%', 'padding': '15px', 'border': f'2px solid {colors["success"]}', 'borderRadius': '5px'})
    ], style={'display': 'flex', 'justifyContent': 'space-between'})
    
    # Create gauge chart
    fig = go.Figure()
    
    # Add RF model gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=rf_pred,
        domain={'x': [0, 0.45], 'y': [0, 1]},
        delta={'reference': 2.0, 'increasing': {'color': colors['success']}, 'decreasing': {'color': colors['danger']}},
        gauge={
            'axis': {'range': [0, 4], 'tickwidth': 1, 'tickcolor': colors['dark']},
            'bar': {'color': colors['info']},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': colors['dark'],
            'steps': [
                {'range': [0, 1], 'color': '#ff9999'},
                {'range': [1, 2], 'color': '#ffcc99'},
                {'range': [2, 3], 'color': '#99ccff'},
                {'range': [3, 4], 'color': '#99ff99'}
            ],
            'threshold': {
                'line': {'color': colors['dark'], 'width': 4},
                'thickness': 0.75,
                'value': rf_pred
            }
        },
        title={'text': "Random Forest Model", 'font': {'size': 16}}
    ))
    
    # Add DL model gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=dl_pred,
        domain={'x': [0.55, 1], 'y': [0, 1]},
        delta={'reference': 2.0, 'increasing': {'color': colors['success']}, 'decreasing': {'color': colors['danger']}},
        gauge={
            'axis': {'range': [0, 4], 'tickwidth': 1, 'tickcolor': colors['dark']},
            'bar': {'color': colors['success']},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': colors['dark'],
            'steps': [
                {'range': [0, 1], 'color': '#ff9999'},
                {'range': [1, 2], 'color': '#ffcc99'},
                {'range': [2, 3], 'color': '#99ccff'},
                {'range': [3, 4], 'color': '#99ff99'}
            ],
            'threshold': {
                'line': {'color': colors['dark'], 'width': 4},
                'thickness': 0.75,
                'value': dl_pred
            }
        },
        title={'text': "Deep Learning Model", 'font': {'size': 16}}
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        title={
            'text': "Grade Prediction Comparison",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        annotations=[
            dict(
                text=f"{rf_letter}",
                x=0.225,
                y=0.4,
                showarrow=False,
                font=dict(size=28)
            ),
            dict(
                text=f"{dl_letter}",
                x=0.775,
                y=0.4,
                showarrow=False,
                font=dict(size=28)
            ),
            dict(
                text="F (0.0)      D (1.0)      C (2.0)      B (3.0)      A (4.0)",
                x=0.5,
                y=0,
                showarrow=False,
                font=dict(size=14)
            )
        ]
    )
    
    return prediction_output, fig

# Callback for model comparison graph
@app.callback(
    Output('model-comparison-graph', 'figure'),
    [Input('predict-button', 'n_clicks')]
)
def update_model_comparison(n_clicks):
    # Calculate predictions on test data
    X, X_scaled = preprocess_data(test_data.drop('GradeClass', axis=1) if 'GradeClass' in test_data.columns else test_data)
    
    # Get actual values if available
    y_actual = test_data['GradeClass'].values if 'GradeClass' in test_data.columns else None
    
    # Make predictions
    if model_1 is not None:
        y_pred_rf = model_1.predict(X)
    else:
        y_pred_rf = np.array([2.0] * len(X))
    
    if model_2 is not None:
        y_pred_dl = model_2.predict(X_scaled)
        # Ensure it's flattened to 1D array
        if y_pred_dl.ndim > 1:
            y_pred_dl = y_pred_dl.flatten()
    else:
        y_pred_dl = np.array([2.0] * len(X))
    
    # Create DataFrame for plotting
    comparison_df = pd.DataFrame({
        'Student': [f'Student {i+1}' for i in range(len(X))],
        'Random Forest': y_pred_rf,
        'Deep Learning': y_pred_dl
    })
    
    if y_actual is not None:
        comparison_df['Actual'] = y_actual
    
    # Melt the DataFrame for easier plotting
    melted_df = pd.melt(comparison_df, id_vars=['Student'], value_vars=['Random Forest', 'Deep Learning'] + (['Actual'] if y_actual is not None else []),
                        var_name='Model', value_name='Grade')
    
    # Create the figure
    fig = px.bar(melted_df, x='Student', y='Grade', color='Model', barmode='group',
                title='Model Prediction Comparison',
                labels={'Grade': 'Predicted Grade', 'Student': 'Student ID'},
                color_discrete_map={
                    'Random Forest': colors['info'],
                    'Deep Learning': colors['success'],
                    'Actual': colors['dark']
                })
    
    # Add grade letter annotations
    for i, model in enumerate(['Random Forest', 'Deep Learning'] + (['Actual'] if y_actual is not None else [])):
        model_data = melted_df[melted_df['Model'] == model]
        for j, (_, row) in enumerate(model_data.iterrows()):
            fig.add_annotation(
                x=row['Student'],
                y=row['Grade'],
                text=get_letter_grade(row['Grade']),
                showarrow=False,
                yshift=10,
                xshift=(-15 + 15*i),
                font=dict(size=10, color='white')
            )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Student',
        yaxis_title='Grade (0.0-4.0)',
        yaxis=dict(range=[0, 4.2]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=400,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig

# Callback for feature importance
@app.callback(
    Output('feature-importance', 'figure'),
    [Input('predict-button', 'n_clicks')]
)
def update_feature_importance(n_clicks):
    # Get feature names
    feature_names = list(test_data.drop('GradeClass', axis=1).columns) if 'GradeClass' in test_data.columns else list(test_data.columns)
    
    # Generate feature importance for Random Forest
    if model_1 is not None and hasattr(model_1, 'feature_importances_'):
        importances = model_1.feature_importances_
    else:
        # Create mock data if model is not available
        importances = np.array([0.25, 0.20, 0.15, 0.15, 0.15, 0.10])
        if len(importances) != len(feature_names):
            importances = np.random.rand(len(feature_names))
            importances = importances / importances.sum()
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Create the figure
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                title='Random Forest Feature Importance',
                labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'},
                color='Importance',
                color_continuous_scale=px.colors.sequential.Blues)
    
    # Update layout
    fig.update_layout(
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=400,
        margin=dict(l=20, r=20, t=80, b=20),
        yaxis=dict(autorange="reversed")
    )
    
    return fig

# Run the application
if __name__ == '__main__':
    app.run(debug=True)