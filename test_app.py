"""Optional automated check of the Streamlit app (run with:  python test_app.py).
It simulates a user and compares the app's output with the model's own prediction."""
import pickle
import pandas as pd
from streamlit.testing.v1 import AppTest

art = {n: pickle.load(open(f"models/{n}.pkl", "rb")) for n in ("preprocessor", "selector", "model", "feature_info")}
info = art["feature_info"]


def model_prediction(solar):
    row = pd.DataFrame([info["defaults"]])
    row["Solar_Generation_kWh"] = solar
    features = art["selector"].transform(art["preprocessor"].transform(row[info["feature_cols"]]))
    return float(art["model"].predict(features)[0])


def shown_prediction(at):
    return float([m.value for m in at.metric if m.label == "Predicted energy saved"][0].split()[0])


at = AppTest.from_file("app.py", default_timeout=60).run()
assert not at.exception

for solar in (15.0, 38.6, 60.0):
    at.number_input[0].set_value(solar).run()
    at.button[0].click().run()
    assert not at.exception
    assert abs(shown_prediction(at) - model_prediction(solar)) < 0.01
    print(f"solar {solar:5.1f} kWh -> app shows {shown_prediction(at):.2f} kWh: OK")

at.radio[0].set_value("Calculate it from sunlight (irradiance)").run()
at.number_input[0].set_value(800.0)
at.number_input[1].set_value(300.0)
at.run()
at.button[0].click().run()
assert abs(shown_prediction(at) - model_prediction(55.0)) < 0.01
print("beam 800 + diffuse 300 (solar 55.0 kWh): OK")
print("All app checks passed.")
