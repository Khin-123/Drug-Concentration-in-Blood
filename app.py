from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from scipy import signal

app = Flask(__name__)

# Real-world approximate PK values (used both in backend + sent to frontend)
DRUG_DB = {
    "Aspirin":     {"ka": 1.5,  "ke": 0.23, "dose": 325,   "interval": 4,  "toxic": 200,  "min_eff": 10},
    "Caffeine":    {"ka": 2.0,  "ke": 0.12, "dose": 100,   "interval": 6,  "toxic": 150,  "min_eff": 10},
    "Theophylline":{"ka": 1.2,  "ke": 0.08, "dose": 200,   "interval": 8,  "toxic": 20,   "min_eff": 10},
    "Digoxin":     {"ka": 0.8,  "ke": 0.02, "dose": 0.25,  "interval": 24, "toxic": 2.0,  "min_eff": 0.5},
    "Custom":      {"ka": 1.2,  "ke": 0.3,  "dose": 100,   "interval": 8,  "toxic": 150,  "min_eff": 30}
}

def generate_plot(params):
    t = np.linspace(0, 48, 1000)

    # Transfer function:   ka / ( (s + ka)(s + ke) )
    num = [params['ka']]
    den = [1, params['ka'] + params['ke'], params['ka'] * params['ke']]
    sys = signal.TransferFunction(num, den)

    # --- Impulse train (oral doses / pills) ---
    u_pill = np.zeros_like(t)
    for d in range(0, 48, params['interval']):
        if d == params['interval'] and params['missed'] > 0:
            # delayed second dose
            time_point = d + params['missed']
        else:
            time_point = d

        idx = np.argmin(np.abs(t - time_point))
        if idx < len(u_pill):           # safety
            u_pill[idx] = params['dose']

    _, y_pill, _ = signal.lsim(sys, U=u_pill, T=t)

    # --- Constant IV infusion ---
    u_iv = np.full_like(t, params['dose'] / params['interval'])
    _, y_iv, _ = signal.lsim(sys, U=u_iv, T=t)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(t, y_pill, label="Oral doses (impulse train)", color='#007bff', linewidth=2)
    plt.plot(t, y_iv,   label="Constant IV infusion",     color='#28a745', linestyle='--')

    plt.axhline(y=params['toxic'],   color='#dc3545', linestyle=':', label="Toxic threshold")
    plt.axhline(y=params['min_eff'], color='#ffc107', linestyle=':', label="Minimum effective")

    plt.fill_between(t, params['min_eff'], params['toxic'], color='#28a745', alpha=0.08)

    plt.title(f"Pharmacokinetics – {params['name']}")
    plt.xlabel("Time (hours)")
    plt.ylabel("Plasma concentration (mg/L)")
    plt.legend()
    plt.grid(True, alpha=0.15)

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()

    # Simple safety evaluation
    max_conc = np.max(y_pill)
    late_conc = np.min(y_pill[int(len(t)*0.25):])   # after first few hours
    status = "SAFE" if (max_conc < params['toxic'] and late_conc > params['min_eff']) else "RISKY"

    return plot_url, status, round(max_conc, 2)


@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    status = None
    max_c = None
    selected_drug = "Aspirin"           # default shown drug

    if request.method == 'POST':
        selected_drug = request.form.get('drug_select', "Custom")

        if selected_drug in DRUG_DB and selected_drug != "Custom":
            p = DRUG_DB[selected_drug].copy()
            p['name'] = selected_drug
        else:
            # Custom
            p = {
                "ka":       float(request.form.get('ka', 1.2)),
                "ke":       float(request.form.get('ke', 0.3)),
                "dose":     float(request.form.get('dose', 100)),
                "interval": int(float(request.form.get('interval', 8))),
                "toxic":    float(request.form.get('toxic', 150)),
                "min_eff":  float(request.form.get('min_eff', 30)),
                "name":     "Custom Drug"
            }

        try:
            p['missed'] = float(request.form.get('missed', 0))
        except:
            p['missed'] = 0.0

        plot_url, status, max_c = generate_plot(p)

    # Always send the list of drug names (as list!)
    drug_list = list(DRUG_DB.keys())

    return render_template(
        'index.html',
        plot_url=plot_url,
        status=status,
        max_c=max_c,
        drugs=drug_list,           # ← fixed: list, not dict_keys
        selected=selected_drug     # ← key name matches template
    )


if __name__ == '__main__':
    app.run(debug=True)