from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from scipy import signal

app = Flask(__name__)

# Drug defaults for quick select
DRUG_DB = {
    "Aspirin": {
        "ka": 1.5, "ke": 0.23, "dose": 325, "interval": 4,
        "toxic": 300, "min_eff": 30,
        "Vd": 10, "F": 1.0, "scale": 1
    },
    "Caffeine": {
        "ka": 2.0, "ke": 0.12, "dose": 100, "interval": 6,
        "toxic": 50, "min_eff": 3,
        "Vd": 42, "F": 1.0, "scale": 1
    },
    "Theophylline": {
        "ka": 1.2, "ke": 0.08, "dose": 200, "interval": 8,
        "toxic": 20, "min_eff": 5,
        "Vd": 35, "F": 1.0, "scale": 1
    },
    "Digoxin": {
        "ka": 0.8, "ke": 0.02, "dose": 0.25, "interval": 24,
        "toxic": 2.0, "min_eff": 0.3,
        "Vd": 490, "F": 0.7, "scale": 1000
    },
    "Custom": {
        "ka": 1.2, "ke": 0.3, "dose": 100, "interval": 8,
        "toxic": 50, "min_eff": 5,
        "Vd": 40, "F": 1.0, "scale": 1
    }
}


def parse_float(form, key, default):
    try:
        return float(form.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def generate_plot(params):
    t = np.linspace(0, 48, 1000)
    dt = t[1] - t[0]
    scale = params.get("scale", 1)

    # Oral transfer function: F * ka / Vd / ((s + ka)(s + ke))
    num_oral = [params["F"] * params["ka"] / params["Vd"]]
    den_oral = [1, params["ka"] + params["ke"], params["ka"] * params["ke"]]
    sys_oral = signal.TransferFunction(num_oral, den_oral)

    # IV transfer function: 1 / Vd / (s + ke)
    num_iv = [1 / params["Vd"]]
    den_iv = [1, params["ke"]]
    sys_iv = signal.TransferFunction(num_iv, den_iv)

    # Impulse train for oral doses
    u_pill = np.zeros_like(t)
    interval = max(1, int(params["interval"]))
    for d in range(0, 48, interval):
        time_point = d
        if d == interval and params["missed"] > 0:
            time_point = d + params["missed"]

        idx = np.argmin(np.abs(t - time_point))
        if idx < len(u_pill):
            # Approximate bolus as an impulse with area equal to dose.
            u_pill[idx] = params["dose"] / dt

    _, y_pill, _ = signal.lsim(sys_oral, U=u_pill, T=t)

    # Constant IV infusion (for comparison)
    u_iv = np.full_like(t, params["dose"] / interval)
    _, y_iv, _ = signal.lsim(sys_iv, U=u_iv, T=t)

    y_pill *= scale
    y_iv *= scale
    toxic_scaled = params["toxic"] * scale
    min_eff_scaled = params["min_eff"] * scale

    plt.figure(figsize=(10, 5))
    plt.plot(t, y_pill, label="Oral doses (impulse train)", color="#007bff", linewidth=2)
    plt.plot(t, y_iv, label="Constant IV infusion", color="#28a745", linestyle="--")
    plt.axhline(y=toxic_scaled, color="#dc3545", linestyle=":", label="Toxic threshold")
    plt.axhline(y=min_eff_scaled, color="#ffc107", linestyle=":", label="Minimum effective")
    plt.fill_between(t, min_eff_scaled, toxic_scaled, color="#28a745", alpha=0.08)
    plt.title(f"Pharmacokinetics - {params['name']}")
    plt.xlabel("Time (hours)")
    unit = "(ng/mL)" if scale > 1 else "(mg/L)"
    plt.ylabel(f"Plasma concentration {unit}")
    plt.legend()
    plt.grid(True, alpha=0.15)

    img = io.BytesIO()
    plt.savefig(img, format="png", dpi=100, bbox_inches="tight")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf-8")
    plt.close()

    max_conc = float(np.max(y_pill))
    if max_conc > toxic_scaled:
        status = "TOXIC"
    elif max_conc < min_eff_scaled:
        status = "SUBTHERAPEUTIC"
    else:
        status = "SAFE"

    return plot_url, status, round(max_conc, 2)


@app.route("/", methods=["GET", "POST"])
def index():
    plot_url = None
    status = None
    max_c = None

    selected_drug = "Aspirin"
    base_defaults = DRUG_DB[selected_drug]
    form_values = {
        "ka": base_defaults["ka"],
        "ke": base_defaults["ke"],
        "vd": base_defaults["Vd"],
        "dose": base_defaults["dose"],
        "interval": base_defaults["interval"],
        "missed": 0,
        "toxic": base_defaults["toxic"],
        "min_eff": base_defaults["min_eff"],
    }

    if request.method == "POST":
        selected_drug = request.form.get("drug_select", "Custom")
        if selected_drug not in DRUG_DB:
            selected_drug = "Custom"

        base = DRUG_DB[selected_drug]
        interval = max(1, int(parse_float(request.form, "interval", base["interval"])))

        p = {
            "ka": parse_float(request.form, "ka", base["ka"]),
            "ke": parse_float(request.form, "ke", base["ke"]),
            "dose": parse_float(request.form, "dose", base["dose"]),
            "interval": interval,
            "toxic": parse_float(request.form, "toxic", base["toxic"]),
            "min_eff": parse_float(request.form, "min_eff", base["min_eff"]),
            "Vd": parse_float(request.form, "vd", base["Vd"]),
            "F": base["F"],
            "scale": base["scale"],
            "name": selected_drug,
            "missed": parse_float(request.form, "missed", 0),
        }

        form_values = {
            "ka": p["ka"],
            "ke": p["ke"],
            "vd": p["Vd"],
            "dose": p["dose"],
            "interval": p["interval"],
            "missed": p["missed"],
            "toxic": p["toxic"],
            "min_eff": p["min_eff"],
        }

        plot_url, status, max_c = generate_plot(p)

    return render_template(
        "index.html",
        plot_url=plot_url,
        status=status,
        max_c=max_c,
        drugs=list(DRUG_DB.keys()),
        selected=selected_drug,
        form_values=form_values,
    )


if __name__ == "__main__":
    app.run(debug=True)
