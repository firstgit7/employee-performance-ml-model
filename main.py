from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import joblib

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Load dataset
df = pd.read_csv("Datasets/garments_worker_productivity.csv")

# Prepare features and target
features = ['quarter', 'department', 'day', 'team', 'targeted_productivity', 'smv', 'wip', 'over_time', 'incentive', 'idle_time', 'idle_men', 'no_of_style_change', 'no_of_workers']
X = pd.get_dummies(df[features])
y = df['actual_productivity']

# Train model
model = RandomForestRegressor()
model.fit(X, y)
joblib.dump(model, "model.pkl")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    fig = px.bar(df, x='department', y='actual_productivity', title='Actual Productivity by Department')
    graph_html = fig.to_html(full_html=False)
    return templates.TemplateResponse("about.html", {"request": request, "graph_html": graph_html})

@app.get("/predict", response_class=HTMLResponse)
async def predict_form(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, 
                  quarter: str = Form(...), 
                  department: str = Form(...), 
                  day: str = Form(...), 
                  team: int = Form(...), 
                  targeted_productivity: float = Form(...), 
                  smv: float = Form(...), 
                  wip: float = Form(...), 
                  over_time: float = Form(...), 
                  incentive: float = Form(...), 
                  idle_time: float = Form(...), 
                  idle_men: float = Form(...), 
                  no_of_style_change: float = Form(...), 
                  no_of_workers: float = Form(...)):
    # Prepare input data
    input_data = pd.DataFrame([{
        'quarter': quarter,
        'department': department,
        'day': day,
        'team': team,
        'targeted_productivity': targeted_productivity,
        'smv': smv,
        'wip': wip,
        'over_time': over_time,
        'incentive': incentive,
        'idle_time': idle_time,
        'idle_men': idle_men,
        'no_of_style_change': no_of_style_change,
        'no_of_workers': no_of_workers
    }])
    input_data = pd.get_dummies(input_data)
    # Align columns with training data
    missing_cols = set(X.columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    input_data = input_data[X.columns]
    # Predict
    prediction = model.predict(input_data)[0]
    return templates.TemplateResponse("predict.html", {"request": request, "prediction": prediction})
