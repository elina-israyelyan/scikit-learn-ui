import base64
import io

import dash
import pandas as pd
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
from sklearn.linear_model import LinearRegression, LogisticRegression

app = dash.Dash(__name__)

# Define CSS styles
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define app layout
app.layout = html.Div([
    html.H1("Model Selection and Data Upload"),

    # Dropdown for model selection
    html.Div([
        html.Label("Select a model:"),
        dcc.Dropdown(
            id='model-dropdown',
            options=[
                {'label': 'Linear Regression', 'value': 'linear_regression'},
                {'label': 'Logistic Regression', 'value': 'logistic_regression'},
            ],
            value=None
        ),
    ], style={'margin-bottom': '20px'}),

    # File upload component
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px 0'
        },
        multiple=False
    ),

    # Container to display form fields dynamically
    html.Div(id='form-fields'),
    html.Button('Train Model & Infer', id='train-button', style={'display': 'none'}),
    html.Div(id='inference-box')

])


# Function to train linear regression model


def train_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


# Function to train logistic regression model
def train_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model


# Callback to dynamically generate form fields based on uploaded data
@app.callback(
    [Output('form-fields', 'children'),
     Output('train-button', 'style')
     ],
    Input('upload-data', 'contents'),
    [State('upload-data', 'filename'),
     State('upload-data', 'last_modified')]
)
def update_form_fields(contents, filename, last_modified):
    if contents is None:
        raise PreventUpdate

    # Read the uploaded file as a pandas DataFrame
    content_type, content_string = contents.split(',')
    df = pd.read_csv(io.StringIO(base64.b64decode(content_string).decode('utf-8')))
    df.pop('y')
    # Train model based on selected model type

    # Generate form fields based on the columns of the DataFrame
    # Generate form fields based on the columns of the DataFrame
    form_fields = [
        html.Div([
            html.Label(column),
            dcc.Input(id=f'{column}-input', type='text', placeholder=f'Enter {column}...')
        ]) for column in df.columns
    ]
    form_fields_container = html.Div([html.H2('Fill in the following fields:'), *form_fields])

    # Concatenate all form fields into a single list
    button_style = {'display': 'block'}

    return form_fields_container, button_style


# Callback to train the model when the train button is clicked
@app.callback(
    Output('inference-box', 'children'),
    [Input('train-button', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('model-dropdown', 'value'),
     State('form-fields', 'children')]
)
def train_model(n_clicks, contents, selected_model, form_fields):
    if n_clicks is None:
        raise PreventUpdate
    content_type, content_string = contents.split(',')
    df = pd.read_csv(io.StringIO(base64.b64decode(content_string).decode('utf-8')))
    y_column = df.pop('y')

    if selected_model == 'linear_regression':
        model = train_linear_regression(df, y_column)
    elif selected_model == 'logistic_regression':
        model = train_logistic_regression(df, y_column)
    else:
        raise ValueError("Invalid model selection")


    # Extract data from form fields
    data = {}
    for child in form_fields['props']['children']:
        if child['type'] == 'Div':
            prop= child['props']['children']
            if prop[0].get('type') == 'Label':
                label = prop[0]['props']['children']
                value = prop[1]["props"]["value"]
                data[label] = value
    print(data)
    df = pd.Series(data).to_frame().T
    print(df)
    # Your model training logic goes here
    # Read the uploaded file and train the model based on selected model type
    # Return the trained model

    inference_result = model.predict(df)

    # Return the inference result
    return inference_result


if __name__ == '__main__':
    app.run_server(debug=True)
