
from flask import Flask, request, jsonify, render_template
import joblib
import csv
import os
from datetime import datetime

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
HISTORY_PATH = os.path.join(BASE_DIR, "exam_history.csv")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        "model.pkl was not found. First run: python train_model.py"
    )

model = joblib.load(MODEL_PATH)


def predict_result(study_hours):
    raw = model.predict([[float(study_hours)]])[0]
    value = str(raw).strip().upper()

    if value in ("1", "1.0", "TRUE") or "PASS" in value:
        return "PASS"

    if value in ("0", "0.0", "FALSE") or "FAIL" in value:
        return "FAIL"

    return str(raw)


def save_history(student_name, study_hours, prediction):
    exists = (
        os.path.isfile(HISTORY_PATH)
        and os.path.getsize(HISTORY_PATH) > 0
    )

    with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "student_name",
                "study_hours",
                "prediction",
                "timestamp"
            ]
        )

        if not exists:
            writer.writeheader()

        writer.writerow({
            "student_name": student_name,
            "study_hours": float(study_hours),
            "prediction": prediction,
            "timestamp": datetime.now().astimezone().isoformat(
                timespec="seconds"
            )
        })


def get_name(data):
    name = str(data.get("student_name", "")).strip()

    if not name:
        raise ValueError(
            "'student_name' is required and cannot be empty."
        )

    return name


def get_hours(data, key):
    value = data.get(key)

    if value is None or str(value).strip() == "":
        raise ValueError(f"'{key}' is required.")

    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number.")

    if number < 0:
        raise ValueError(f"'{key}' cannot be negative.")

    return number


def records():
    if not os.path.isfile(HISTORY_PATH):
        return []

    with open(HISTORY_PATH, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/predict")
def predict():
    try:
        data = request.get_json(silent=True) or {}

        name = get_name(data)
        hours = get_hours(data, "study_hours")
        result = predict_result(hours)

        save_history(name, hours, result)

        return jsonify({
            "student_name": name,
            "study_hours": hours,
            "prediction": result
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


@app.post("/simulate")
def simulate():
    try:
        data = request.get_json(silent=True) or {}

        name = get_name(data)
        current = get_hours(data, "current_study_hours")
        additional = get_hours(data, "additional_study_hours")

        original = predict_result(current)
        total = current + additional
        simulated = predict_result(total)

        save_history(name, current, original)
        save_history(name, total, simulated)

        return jsonify({
            "student_name": name,
            "current_study_hours": current,
            "additional_study_hours": additional,
            "simulated_study_hours": total,
            "original_prediction": original,
            "simulated_prediction": simulated,
            "prediction_changed": original != simulated
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        app.logger.exception("Simulation failed")

        return jsonify({
            "error": "Simulation failed",
            "details": str(e)
        }), 500


@app.post("/passing-hours")
def passing_hours():
    try:
        data = request.get_json(silent=True) or {}
        name = get_name(data)

        found = None

        for i in range(49):  # 0, 0.5, 1, ... 24 hours
            hours = i * 0.5

            if predict_result(hours) == "PASS":
                found = hours
                break

        if found is None:
            return jsonify({
                "student_name": name,
                "first_tested_passing_hours": None,
                "message": (
                    "No PASS prediction found from 0 to 24 hours. "
                    "This is not a guaranteed passing requirement."
                )
            })

        save_history(name, found, "PASS")

        return jsonify({
            "student_name": name,
            "first_tested_passing_hours": found,
            "message": (
                "This is the first value predicted PASS by this model, "
                "not a guaranteed passing requirement."
            )
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        app.logger.exception("Passing-hours search failed")

        return jsonify({
            "error": "Passing-hours search failed",
            "details": str(e)
        }), 500


@app.post("/suggestions")
def suggestions():
    try:
        data = request.get_json(silent=True) or {}

        name = get_name(data)
        hours = get_hours(data, "study_hours")
        result = predict_result(hours)

        if result == "PASS":
            advice = [
                "Keep following your current study routine.",
                "Review difficult topics to strengthen your understanding.",
                "Practice sample questions before the exam.",
                "Get enough sleep before examination day."
            ]
        else:
            advice = [
                "Create a daily study schedule and follow it consistently.",
                "Review lessons you find difficult.",
                "Study in focused sessions with short breaks.",
                "Practice sample questions and review your mistakes.",
                "Consider increasing your study time gradually."
            ]

        return jsonify({
            "student_name": name,
            "study_hours": hours,
            "prediction": result,
            "suggestions": advice,
            "note": (
                "These are general study tips. Predictions are estimates, "
                "not guarantees of exam results."
            )
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        app.logger.exception("Suggestions failed")

        return jsonify({
            "error": "Could not generate suggestions",
            "details": str(e)
        }), 500


@app.get("/history")
def history():
    rows = records()

    wanted = request.args.get("student_name", "").strip().casefold()

    if wanted:
        rows = [
            r for r in rows
            if r.get("student_name", "").casefold() == wanted
        ]

    return jsonify({
        "count": len(rows),
        "history": rows
    })


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
