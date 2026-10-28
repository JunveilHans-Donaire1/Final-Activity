# Activity 3 — Student Exam Result Prediction System

## You said you do not have model.pkl
This project includes `train_model.py` to create a local demonstration `model.pkl`.
Because the original Activity 1 model is missing, the script uses synthetic example
data. It is not your instructor's original model and does not establish a real
minimum passing-hours requirement. If you recover your original model, use it instead.

## Run offline
First make sure Python, Flask, joblib and scikit-learn are installed. Installing packages
for the first time may need internet; running the app after installation does not.

In PowerShell, from this folder:

```powershell
python -m pip install -r requirements.txt
python train_model.py
python app.py
```

Leave the terminal running. The local API is at `http://127.0.0.1:5000`.

## Test using PowerShell

Original prediction:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/predict -ContentType "application/json" -Body '{"student_name":"Juan Dela Cruz","study_hours":3}'
```

Simulation:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/simulate -ContentType "application/json" -Body '{"student_name":"Juan Dela Cruz","current_study_hours":3,"additional_study_hours":4}'
```

Passing-hours estimator:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/passing-hours -ContentType "application/json" -Body '{"student_name":"Juan Dela Cruz"}'
```

History:
Open `http://127.0.0.1:5000/history` in your browser.

The CSV file `exam_history.csv` is created automatically when a prediction is saved.
