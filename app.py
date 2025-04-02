from flask import Flask, request, render_template
import sys
from io import StringIO
import numpy as np
from itertools import combinations
import os

app = Flask(__name__)

# Your original code (unchanged) goes here
standard_resistors = np.array([
    10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82,
    100, 120, 150, 180, 220, 270, 330, 390, 470, 560, 680, 820,
    1000, 1200, 1500, 1800, 2200, 2700, 3300, 3900, 4700, 5600, 6800, 8200,
    10000, 12000, 15000, 18000, 22000, 27000, 33000, 39000, 47000, 56000, 68000, 82000, 100000
])

standard_capacitors = np.array([
    1e-12, 2.2e-12, 4.7e-12, 10e-12, 22e-12, 47e-12,
    100e-12, 220e-12, 470e-12, 1e-9, 2.2e-9, 4.7e-9, 10e-9, 22e-9, 47e-9,
    100e-9, 220e-9, 470e-9, 1e-6, 2.2e-6, 4.7e-6, 10e-6, 22e-6, 47e-6, 100e-6, 220e-6, 470e-6, 1000e-6, 2200e-6
])

transistors = [
    ("BC547", 100, 45, 110, 300),
    ("2N3904", 200, 40, 100, 300),
    ("2N2222", 800, 40, 100, 250),
    ("BC337", 800, 45, 100, 100),
    ("BD139", 1500, 80, 40, 50),
    ("TIP31", 3000, 100, 25, 3),
    ("2N3055", 15000, 60, 20, 2)
]

def get_input_value(prompt, optional=False, value=None):
    if value is None and optional:
        return None
    elif value is None and not optional:
        raise ValueError(f"{prompt} is required.")
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"Invalid input for {prompt}. Please enter a valid number.")

# [Rest of your functions unchanged: find_best_resistor, find_best_capacitor, etc.]
def find_best_resistor(target, values):
    if target <= 0:
        return "N/A (invalid target)", 1000
    if target < 10:
        return "N/A (target too small)", 10
    best_single = values[np.argmin(np.abs(values - target))]
    single_error = abs(best_single - target) / target
    if single_error < 0.2:
        result = best_single
        comb_str = f"{result/1000:.1f} kΩ" if result >= 1000 else f"{result:.1f} Ω"
        return f"{comb_str} (error: {single_error*100:.1f}%)", result
    min_error = single_error
    best_match = ([best_single], 'single', best_single)
    for num in range(2, 4):
        for comb in combinations(values, num):
            series_result = sum(comb)
            series_error = abs(series_result - target) / target
            if series_error < min_error:
                min_error = series_error
                best_match = (comb, 'series', series_result)
            if all(r > 0 for r in comb):
                parallel_result = 1 / sum(1 / r for r in comb)
                parallel_error = abs(parallel_result - target) / target
                if parallel_error < min_error:
                    min_error = parallel_error
                    best_match = (comb, 'parallel', parallel_result)
    comb, mode, result = best_match
    formatted_values = [f"{v/1000:.1f} kΩ" if v >= 1000 else f"{v:.1f} Ω" for v in comb]
    comb_str = " + " if mode == 'series' else " || " if mode == 'parallel' else formatted_values[0]
    result_str = f"{result/1000:.2f} kΩ" if result >= 1000 else f"{result:.2f} Ω"
    return f"{comb_str.join(formatted_values)} ({mode}, {result_str}, error: {min_error*100:.1f}%)", result

def find_best_capacitor(target, values):
    if target <= 0:
        return "N/A (invalid target)", 1e-6
    if target < 1e-12:
        return "N/A (target too small)", 1e-12
    best_single = values[np.argmin(np.abs(values - target))]
    single_error = abs(best_single - target) / target
    if single_error < 0.2:
        result = best_single
        comb_str = format_capacitance(result)
        return f"{comb_str} (error: {single_error*100:.1f}%)", result
    min_error = single_error
    best_match = ([best_single], 'single', best_single)
    for comb in combinations(values, 2):
        if all(c > 0 for c in comb):
            series_result = 1 / sum(1 / c for c in comb)
            series_error = abs(series_result - target) / target
            if series_error < min_error:
                min_error = series_error
                best_match = (comb, 'series', series_result)
        parallel_result = sum(comb)
        parallel_error = abs(parallel_result - target) / target
        if parallel_error < min_error:
            min_error = parallel_error
            best_match = (comb, 'parallel', parallel_result)
    comb, mode, result = best_match
    formatted_values = [format_capacitance(v) for v in comb]
    comb_str = " || " if mode == 'series' else " + " if mode == 'parallel' else formatted_values[0]
    result_str = format_capacitance(result)
    return f"{comb_str.join(formatted_values)} ({mode}, {result_str}, error: {min_error*100:.1f}%)", result

def format_resistance(value):
    if value >= 1e6:
        return f"{value/1e6:.2f} MΩ"
    elif value >= 1000:
        return f"{value/1000:.2f} kΩ"
    else:
        return f"{value:.2f} Ω"

def format_capacitance(value):
    if value >= 1e-6:
        return f"{value*1e6:.2f} µF"
    elif value >= 1e-9:
        return f"{value*1e9:.2f} nF"
    else:
        return f"{value*1e12:.2f} pF"

def parallel(r1, r2):
    if r1 <= 0 or r2 <= 0:
        return max(r1, r2)
    return (r1 * r2) / (r1 + r2)

def capacitor_reactance(c, f):
    if c <= 0 or f <= 0:
        return float('inf')
    return 1 / (2 * np.pi * f * c)

def recommend_transistor(Ic_mA, Vce, Av, BW=None):
    required_Ic = Ic_mA * 1.5
    required_Vce = Vce * 1.5
    suitable = [t for t in transistors if required_Ic <= 0.7 * t[1] and required_Vce <= 0.7 * t[2]]
    if not suitable:
        return max(transistors, key=lambda t: (t[1], t[2]))
    if BW and BW > 0:
        suitable = [t for t in suitable if t[4] >= 20 * BW / 1e6]
    return max(suitable, key=lambda t: t[3]) if Av > 50 else min(suitable, key=lambda t: abs(0.5 - Ic_mA/t[1]) + abs(0.5 - Vce/t[2]))

def amplifier_design(Vcc, Av, BW, Ic=None, Vce=None, Zi=None, RL=None, Rs=None, Vs=None, Beta=None):
    output = StringIO()
    sys.stdout = output

    print("\n===== CE Voltage Divider Bias Amplifier Design Calculator =====")
    
    Vcc = get_input_value("Enter supply voltage (Vcc) in volts", value=Vcc)
    if Vcc <= 0:
        print("Supply voltage must be positive. Using default 12V.")
        Vcc = 12
    Av = get_input_value("Enter desired voltage gain (Av) magnitude (positive number)", value=Av)
    if Av <= 0:
        print("Gain must be positive. Using default 10.")
        Av = 10
    BW = get_input_value("Enter desired low-frequency cutoff in Hz", value=BW)
    if BW <= 0:
        print("Bandwidth must be positive. Using default 20Hz.")
        BW = 20
    
    print("\nOptional parameters (press Enter to use calculated defaults):")
    Ic = get_input_value("Enter desired Ic at Q-point in mA", optional=True, value=Ic)
    Vce = get_input_value("Enter desired Vce at Q-point in volts", optional=True, value=Vce)
    Zi = get_input_value("Enter input impedance (Zi) in ohms", optional=True, value=Zi)
    RL = get_input_value("Enter load resistance (RL) in ohms", optional=True, value=RL)
    Rs = get_input_value("Enter source resistance (Rs) in ohms", optional=True, value=Rs)
    Vs = get_input_value("Enter input signal amplitude (Vs) in volts", optional=True, value=Vs)
    Beta = get_input_value("Enter transistor current gain (Beta)", optional=True, value=Beta)

    Vt = 0.026
    if Beta is None or Beta <= 0:
        Beta = 100
        print(f"Using default Beta = {Beta}")
    if Vce is None or Vce <= 0 or Vce >= Vcc:
        Vce = Vcc / 2
        print(f"Using default Vce = {Vce:.2f} V (Vcc/2)")
    if Ic is None or Ic <= 0:
        Ic_A = min(0.004, Vcc / 3000)
        Ic = Ic_A * 1000
        print(f"Using default Ic = {Ic:.2f} mA")
    else:
        Ic_A = Ic / 1000
    if RL is None or RL <= 0:
        RL = 10000
        print(f"Using default RL = {format_resistance(RL)}")
    if Rs is None or Rs < 0:
        Rs = 600
        print(f"Using default Rs = {format_resistance(Rs)}")

    # [Rest of your amplifier_design function unchanged]
    Ve = 0.1 * Vcc
    Re = Ve / Ic_A
    Ib = Ic_A / Beta
    I_divider = max(10 * Ib, 0.0001)
    Vb = Ve + 0.7
    R2 = Vb / I_divider
    R1 = (Vcc - Vb) / I_divider

    R1_best_str, R1_actual = find_best_resistor(R1, standard_resistors)
    R2_best_str, R2_actual = find_best_resistor(R2, standard_resistors)
    Re_best_str, Re_actual = find_best_resistor(Re, standard_resistors)

    Ic_actual = Ic_A
    Ve_actual = Ic_actual * Re_actual
    Vb_actual = Ve_actual + 0.7
    if Ve_actual > 0.2 * Vcc or Vb_actual > Vcc:
        Ve_actual = 0.1 * Vcc
        Re = Ve_actual / Ic_actual
        Re_best_str, Re_actual = find_best_resistor(Re, standard_resistors)
        Ve_actual = Ic_actual * Re_actual
        Vb_actual = Ve_actual + 0.7
        R2 = Vb_actual / I_divider
        R1 = (Vcc - Vb_actual) / I_divider
        R1_best_str, R1_actual = find_best_resistor(R1, standard_resistors)
        R2_best_str, R2_actual = find_best_resistor(R2, standard_resistors)

    re_ac = Vt / Ic_actual
    Rc = (Vcc - Vce - Ve_actual) / Ic_actual
    Rc_best_str, Rc_actual = find_best_resistor(Rc, standard_resistors)
    Rc_effective = parallel(Rc_actual, RL)
    Vce_actual = Vcc - Ic_actual * Rc_actual - Ve_actual
    if abs(Vce_actual - Vce) / Vce > 0.1:
        print(f"Adjusting Rc to maintain Vce ≈ {Vce:.1f} V (current Vce={Vce_actual:.1f})...")
        Rc = (Vcc - Vce - Ve_actual) / Ic_actual
        Rc_best_str, Rc_actual = find_best_resistor(Rc, standard_resistors)
        Rc_effective = parallel(Rc_actual, RL)
        Vce_actual = Vcc - Ic_actual * Rc_actual - Ve_actual

    Av_with_no_bypass = Rc_effective / (re_ac + Re_actual)
    if BW:
        Zb = Beta * (re_ac + Re_actual)
        R1_parallel_R2 = parallel(R1_actual, R2_actual)
        Zi_actual = parallel(Zb, R1_parallel_R2)
        Zi_with_source = Zi or Zi_actual
        if Rs:
            Zi_with_source = parallel(Zi_with_source, Rs)
        Cin = 1 / (2 * np.pi * BW * Zi_with_source)
        Cout = 1 / (2 * np.pi * BW * RL)
        Ce = 1 / (2 * np.pi * BW * (re_ac / 10))

        Cin_best_str, Cin_actual = find_best_capacitor(Cin, standard_capacitors)
        Cout_best_str, Cout_actual = find_best_capacitor(Cout, standard_capacitors)
        Ce_best_str, Ce_actual = find_best_capacitor(Ce, standard_capacitors)

        Xce_at_BW = capacitor_reactance(Ce_actual, BW)
        Ze_magnitude = parallel(Re_actual, Xce_at_BW) if Xce_at_BW > 0 else Re_actual
        Av_actual = Rc_effective / (re_ac + Ze_magnitude)
        if abs(Av_actual - Av) / Av <= 0.1:
            Av_actual = Av
        elif Av_actual > Av * 1.1:
            print(f"Warning: Calculated gain {Av_actual:.1f} exceeds target {Av:.1f}. Capping at target.")
            Av_actual = Av
    else:
        Zi_actual = None
        Cin_actual = Cout_actual = Ce_actual = None
        Av_actual = Av_with_no_bypass

    Vout_max_pp = 2 * min(Vce_actual - 0.3, Ve_actual - 0.1)
    Vin_max_pp = Vout_max_pp / Av_actual if Av_actual > 0 else 0
    recommended = recommend_transistor(Ic_actual*1000*1.5, Vce_actual*1.5, Av, BW)

    print("\n============== DESIGN RESULTS ==============")
    print(f"\n🔹 Q-Point Parameters:")
    try:
        print(f"  • Collector Current (Ic): {Ic_actual*1000:.2f} mA (target: {Ic:.2f} mA)")
        print(f"  • Collector-Emitter Voltage (Vce): {Vce_actual:.2f} V (target: {Vce:.2f} V)")
        print(f"  • Base Current (Ib): {(Ic_actual/Beta)*1000000:.2f} µA")
        print(f"  • Emitter Voltage (Ve): {Ve_actual:.2f} V")
        print(f"  • Small-signal emitter resistance (re'): {re_ac:.2f} Ω")
    except Exception as e:
        print(f"Error in Q-Point display: {e}")

    print(f"\n🔹 Component Values:")
    print(f"  • Collector Resistor (Rc): {format_resistance(Rc)} (adjusted target)")
    print(f"    ➤ Best Match: {Rc_best_str}")
    print(f"  • Emitter Resistor (Re): {format_resistance(Re)} (target)")
    print(f"    ➤ Best Match: {Re_best_str}")
    print(f"  • Base Resistor R1: {format_resistance(R1)} (target)")
    print(f"    ➤ Best Match: {R1_best_str}")
    print(f"  • Base Resistor R2: {format_resistance(R2)} (target)")
    print(f"    ➤ Best Match: {R2_best_str}")
    
    if BW:
        print(f"\n🔹 Capacitor Values:")
        print(f"  • Input Coupling Capacitor (Cin): {format_capacitance(Cin)} (target)")
        print(f"    ➤ Best Match: {Cin_best_str}")
        print(f"  • Output Coupling Capacitor (Cout): {format_capacitance(Cout)} (target)")
        print(f"    ➤ Best Match: {Cout_best_str}")
        print(f"  • Emitter Bypass Capacitor (Ce): {format_capacitance(Ce)} (target)")
        print(f"    ➤ Best Match: {Ce_best_str}")
    
    print(f"\n🔹 Impedance Parameters:")
    if Zi_actual:
        print(f"  • Input Impedance: {format_resistance(Zi_actual)}")
        if Rs:
            print(f"  • Source Resistance: {format_resistance(Rs)}")
            print(f"  • Total Input Impedance with Source: {format_resistance(Zi_with_source)}")
    print(f"  • Output Impedance: {format_resistance(Rc_actual)}")
    print(f"  • Load Resistance: {format_resistance(RL)}")
    print(f"  • Effective Collector Load: {format_resistance(Rc_effective)}")
    
    print(f"\n🔹 Recommended Transistor:")
    print(f"  • {recommended[0]} (Ic_max={recommended[1]}mA, Vce_max={recommended[2]}V, β≈{recommended[3]}, ft≈{recommended[4]}MHz)")
    
    print(f"\n🔹 Performance Analysis:")
    print(f"  • Target Voltage Gain: {Av:.1f}")
    print(f"  • Actual Expected Gain: {int(round(Av_actual))}")
    print(f"  • Gain without Emitter Bypass: {Av_with_no_bypass:.1f}")
    print(f"  • Maximum Output Voltage Swing: {Vout_max_pp:.2f} Vpp")
    print(f"  • Maximum Input Voltage for Linearity: {Vin_max_pp*1000:.2f} mVpp")
    
    if BW and Cin_actual and Cout_actual:
        print(f"\n🔹 Frequency Response:")
        f_low_input = 1 / (2 * np.pi * Cin_actual * Zi_with_source) if Cin_actual > 0 and Zi_with_source > 0 else float('inf')
        f_low_output = 1 / (2 * np.pi * Cout_actual * RL) if Cout_actual > 0 and RL > 0 else float('inf')
        print(f"  • Input Coupling Cutoff: {f_low_input:.1f} Hz")
        print(f"  • Output Coupling Cutoff: {f_low_output:.1f} Hz")
        
        print(f"  • Emitter Bypass Performance:")
        for freq in [BW/2, BW, BW*2, BW*5]:
            Xce = capacitor_reactance(Ce_actual, freq)
            bypass_effect = re_ac / (re_ac + Xce) if Xce > 0 else 1
            print(f"    ➤ At {freq:.1f} Hz: Xce = {Xce:.1f} Ω, Bypass Effectiveness: {bypass_effect*100:.1f}%")
        
        overall_cutoff = max(f_low_input, f_low_output)
        print(f"  • Overall Low Frequency Cutoff: {overall_cutoff:.1f} Hz (target was {BW} Hz)")
        
        if recommended[4] > 0:
            C_be = 20e-12
            C_bc = 6e-12
            C_in = C_be + C_bc * (1 + Av_actual)
            R_in = Zi_with_source
            f_high = 1 / (2 * np.pi * R_in * C_in) / 1000
            print(f"  • Estimated High Frequency Cutoff: {f_high:.1f} kHz (with Cbc={C_bc*1e12:.0f}pF, Cbe={C_be*1e12:.0f}pF)")
            print(f"  • Estimated Bandwidth: {f_high - overall_cutoff/1000:.1f} kHz")
    
    print("\n=== Design Completed Successfully ===")

    sys.stdout = sys.__stdout__
    return output.getvalue()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            Vcc = request.form.get('Vcc')
            Av = request.form.get('Av')
            BW = request.form.get('BW')
            Ic = request.form.get('Ic') or None
            Vce = request.form.get('Vce') or None
            Zi = request.form.get('Zi') or None
            RL = request.form.get('RL') or None
            Rs = request.form.get('Rs') or None
            Vs = request.form.get('Vs') or None
            Beta = request.form.get('Beta') or None

            result = amplifier_design(Vcc, Av, BW, Ic, Vce, Zi, RL, Rs, Vs, Beta)
            return render_template('index.html', result=result)
        except Exception as e:
            return render_template('index.html', result=f"Error: {str(e)}")
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)