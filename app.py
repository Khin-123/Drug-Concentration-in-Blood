from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import json
from scipy import signal

app = Flask(__name__)

# ==============================
# Updated Drug Database
# ==============================

DRUG_DB = {
    "Theophylline": {
        "ka": 1.0,
        "ke": 0.1,
        "dose": 200,
        "doses_per_day": 4,
        "interval": 16 / 4,
        "toxic": 15,
        "min_eff": 5,
        "Vd": 35,
        "F": 1.0,
        "scale": 1,
    },
    "Antibiotics": {  # generic beta-lactam profile
        "ka": 1.3,
        "ke": 0.5,
        "dose": 500,
        "doses_per_day": 3,
        "interval": 16 / 3,
        "toxic": 30,
        "min_eff": 10,
        "Vd": 20,
        "F": 0.9,
        "scale": 1,
    },
    "Furosemide": {
        "ka": 1.2,
        "ke": 0.35,
        "dose": 40,
        "doses_per_day": 2,
        "interval": 16 / 2,
        "toxic": 6,
        "min_eff": 1,
        "Vd": 12,
        "F": 0.7,
        "scale": 1,
    },
    "Metronidazole": {
        "ka": 1.1,
        "ke": 0.15,
        "dose": 400,
        "doses_per_day": 3,
        "interval": 16 / 3,
        "toxic": 30,
        "min_eff": 8,
        "Vd": 40,
        "F": 1.0,
        "scale": 1,
    },
    "Ciprofloxacin": {
        "ka": 1.5,
        "ke": 0.25,
        "dose": 500,
        "doses_per_day": 3,
        "interval": 16 / 3,
        "toxic": 6,
        "min_eff": 1,
        "Vd": 30,
        "F": 0.8,
        "scale": 1,
    },
    "Custom": {
        "ka": 1.2,
        "ke": 0.3,
        "dose": 100,
        "doses_per_day": 3,
        "interval": 16 / 3,
        "toxic": 30,
        "min_eff": 5,
        "Vd": 40,
        "F": 1.0,
        "scale": 1,
    },
}

SIM_TIME = 16  # 16 hours simulation


def parse_float(form, key, default):
    try:
        return float(form.get(key, default))
    except Exception:
        return float(default)


def _find_subtherapeutic_ranges(times, mask):
    ranges = []
    in_range = False
    start_t = 0.0

    for ti, is_sub in zip(times, mask):
        if is_sub and not in_range:
            in_range = True
            start_t = float(ti)
        elif (not is_sub) and in_range:
            in_range = False
            ranges.append((start_t, float(ti)))

    if in_range and len(times) > 0:
        ranges.append((start_t, float(times[-1])))

    return ranges


def generate_plot(params):
    t = np.linspace(0, SIM_TIME, 1000)
    dt = t[1] - t[0]
    scale = params.get("scale", 1)

    # Transfer functions
    num_oral = [params["F"] * params["ka"] / params["Vd"]]
    den_oral = [1, params["ka"] + params["ke"], params["ka"] * params["ke"]]
    sys_oral = signal.TransferFunction(num_oral, den_oral)

    num_iv = [1 / params["Vd"]]
    den_iv = [1, params["ke"]]
    sys_iv = signal.TransferFunction(num_iv, den_iv)

    # ========================
    # ORAL DOSE TIMING
    # ========================
    doses_per_day = max(1, int(params["doses_per_day"]))
    interval = params["interval"]
    delay2 = params.get("delay2", 0)
    delay3 = params.get("delay3", 0)

    # Support any number of doses/day; keep optional delay controls for dose 2 and 3.
    dose_times = []
    for i in range(doses_per_day):
        time_point = i * interval
        if i == 1:
            time_point += delay2
        elif i == 2:
            time_point += delay3
        dose_times.append(time_point)

    # Impulse train
    u_pill = np.zeros_like(t)
    for time_point in dose_times:
        if 0 <= time_point <= SIM_TIME:
            idx = np.argmin(np.abs(t - time_point))
            u_pill[idx] = params["dose"] / dt

    _, y_pill, _ = signal.lsim(sys_oral, U=u_pill, T=t)

    # ========================
    # IV INFUSION (constant)
    # ========================
    u_iv = np.full_like(t, params["dose"] / interval)
    _, y_iv, _ = signal.lsim(sys_iv, U=u_iv, T=t)

    y_pill *= scale
    y_iv *= scale

    toxic_scaled = params["toxic"] * scale
    min_eff_scaled = params["min_eff"] * scale

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(t, y_pill, label="Oral doses", linewidth=2)
    plt.plot(t, y_iv, linestyle="--", label="Continuous IV")
    plt.axhline(y=toxic_scaled, linestyle=":", label="Toxic")
    plt.axhline(y=min_eff_scaled, linestyle=":", label="Min Effective")
    plt.fill_between(t, min_eff_scaled, toxic_scaled, alpha=0.1)

    plt.title(f"Pharmacokinetics - {params['name']}")
    plt.xlabel("Time (hours)")
    plt.ylabel("Plasma Concentration (mg/L)")
    plt.legend()
    plt.grid(alpha=0.2)

    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    max_conc = float(np.max(y_pill))

    effective_mask = y_pill >= min_eff_scaled
    ever_effective = bool(np.any(effective_mask))

    if ever_effective:
        first_effective_idx = int(np.argmax(effective_mask))
        post_effective_t = t[first_effective_idx:]
        post_effective_sub_mask = y_pill[first_effective_idx:] < min_eff_scaled
        dips_after_effective = bool(np.any(post_effective_sub_mask))
    else:
        post_effective_t = t
        post_effective_sub_mask = y_pill < min_eff_scaled
        dips_after_effective = True

    # Count how many dose intervals never reach minimum effective level.
    ineffective_intervals = 0
    interval_count = 0
    for i, start in enumerate(dose_times):
        end = dose_times[i + 1] if i + 1 < len(dose_times) else SIM_TIME
        if end <= start:
            continue
        mask = (t >= start) & (t < end)
        if not np.any(mask):
            continue
        interval_count += 1
        if float(np.max(y_pill[mask])) < min_eff_scaled:
            ineffective_intervals += 1

    ineffective_ratio = (ineffective_intervals / interval_count) if interval_count else 0.0
    toxic_overshoot = max_conc > toxic_scaled
    many_ineffective_doses = ineffective_ratio >= 0.5

    if toxic_overshoot or many_ineffective_doses:
        status = "RISKY"
    elif ever_effective and (not dips_after_effective):
        status = "SAFE"
    else:
        status = "SUBTHERAPEUTIC"

    messages = []

    if toxic_overshoot:
        messages.append(
            "Warning: Drug concentration exceeds the toxic threshold. This schedule may cause adverse effects."
        )

    if many_ineffective_doses:
        messages.append(
            f"Risky: {ineffective_intervals}/{interval_count} dose intervals never reach minimum effective concentration."
        )

    if not ever_effective:
        messages.append(
            "Warning: Drug levels never reach the minimum effective concentration."
        )

    if ever_effective and dips_after_effective:
        messages.append(
            "Warning: Drug concentration drops below minimum effective after becoming therapeutic."
        )

    if status in ("SUBTHERAPEUTIC", "RISKY"):
        sub_ranges = _find_subtherapeutic_ranges(post_effective_t, post_effective_sub_mask)
        if sub_ranges:
            ranges_text = ", ".join(f"{s:.2f}-{e:.2f} h" for s, e in sub_ranges[:3])
            if len(sub_ranges) > 3:
                ranges_text += ", ..."
            messages.append(f"Subtherapeutic periods: {ranges_text}")

    if delay2 > 0.5:
        messages.append("Note: There is a noticeable delay in the second dose.")

    if delay3 > 0.5:
        messages.append("Note: There is a delay in the third dose.")

    if interval < (SIM_TIME / doses_per_day) * 0.7:
        messages.append("Warning: Doses are very frequent. Monitor for accumulation.")

    if status == "SAFE":
        messages.append("OK: Drug concentration remains within the therapeutic window.")

    return plot_url, status, round(max_conc, 2), messages


@app.route("/", methods=["GET", "POST"])
def index():
    selected_drug = "Theophylline"
    base = DRUG_DB[selected_drug]

    form_values = {
        "ka": base["ka"],
        "ke": base["ke"],
        "Vd": base["Vd"],
        "dose": base["dose"],
        "doses_per_day": base["doses_per_day"],
        "interval": base.get("interval", 16 / base["doses_per_day"]),
        "delay2": 0,
        "delay3": 0,
        "toxic": base["toxic"],
        "min_eff": base["min_eff"],
    }

    plot_url = None
    status = None
    max_c = None
    messages = None

    if request.method == "POST":
        selected_drug = request.form.get("drug_select")
        base = DRUG_DB[selected_drug]

        p = {
            "ka": parse_float(request.form, "ka", base["ka"]),
            "ke": parse_float(request.form, "ke", base["ke"]),
            "dose": parse_float(request.form, "dose", base["dose"]),
            "doses_per_day": int(parse_float(request.form, "doses_per_day", base["doses_per_day"])),
            "interval": parse_float(request.form, "interval", base.get("interval", 16 / base["doses_per_day"])),
            "delay2": parse_float(request.form, "delay2", 0),
            "delay3": parse_float(request.form, "delay3", 0),
            "toxic": parse_float(request.form, "toxic", base["toxic"]),
            "min_eff": parse_float(request.form, "min_eff", base["min_eff"]),
            "Vd": parse_float(request.form, "Vd", base["Vd"]),
            "F": base["F"],
            "scale": base["scale"],
            "name": selected_drug,
        }

        form_values = p
        plot_url, status, max_c, messages = generate_plot(p)

    return render_template(
        "index.html",
        plot_url=plot_url,
        status=status,
        max_c=max_c,
        messages=messages,
        drugs=list(DRUG_DB.keys()),
        selected=selected_drug,
        form_values=form_values,
        drugs_json=json.dumps(DRUG_DB),
    )


if __name__ == "__main__":
    app.run(debug=True)
