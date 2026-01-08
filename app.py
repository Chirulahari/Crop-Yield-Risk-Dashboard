from flask import Flask, render_template, request, redirect
from model.train_predict import run_model
from dashboard_dash import create_dash_app

app = Flask(__name__)

# Global storage for dashboard data
DASH_DATA = {}

# ✅ Create Dash ONCE (empty at start)
dash_app = create_dash_app(app, DASH_DATA)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global DASH_DATA

    file = request.files["dataset"]
    file_path = f"uploads/{file.filename}"
    file.save(file_path)

    # Run model
    DASH_DATA.clear()
    DASH_DATA.update(run_model(file_path))

    # Redirect to dashboard
    return redirect("/dashboard/")


if __name__ == "__main__":
    app.run(debug=True)
