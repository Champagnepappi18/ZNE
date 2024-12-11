# Quantum Circuit Simulation with Zero-Noise Extrapolation (ZNE)

## Project Overview
This project explores the use of **Zero-Noise Extrapolation (ZNE)** to mitigate quantum noise in quantum circuits. By simulating quantum circuits with amplified noise levels and extrapolating to a zero-noise limit, the project demonstrates how to improve computational accuracy and reduce errors caused by quantum noise.

---

## Key Features
- **Quantum Circuit Simulation**:
  - Design and simulate quantum circuits using the Qiskit library.
- **Noise Model Configuration**:
  - Introduce depolarizing noise to mimic real-world quantum hardware behavior.
- **Zero-Noise Extrapolation (ZNE)**:
  - Amplify noise levels, fit noisy results to a polynomial model, and extrapolate to predict zero-noise outcomes.
- **Visualization**:
  - Graph noisy results, linear fits, and zero-noise estimates to demonstrate the effectiveness of ZNE.

---

## Code Details

### 1. Circuit Creation and Simulation
Design quantum circuits using Qiskit and simulate them on the **QASM Simulator**.

```python
from qiskit import QuantumCircuit, Aer, execute

# Create a Quantum Circuit
qc = QuantumCircuit(1, 1)
qc.h(0)  # Apply Hadamard gate
qc.measure(0, 0)  # Measure the qubit

# Run the circuit on the QASM simulator
simulator = Aer.get_backend('qasm_simulator')
job = execute(qc, simulator, shots=1024)
result = job.result()
print("Counts:", result.get_counts())
```

### 2. Noise Model Configuration
Simulate realistic noise using depolarizing noise models.

```python
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error

# Define a noise model
noise_model = NoiseModel()
error = depolarizing_error(0.01, 1)  # Depolarizing noise with 1% error rate
noise_model.add_all_qubit_quantum_error(error, ['h'])  # Apply to Hadamard gate
```

### 3. Noise Amplification
Scale noise levels to generate data for extrapolation.

```python
# Simulate with increasing noise levels
results = []
for scale_factor in [1.0, 1.5, 2.0]:
    scaled_model = noise_model.copy()
    scaled_model.add_all_qubit_quantum_error(depolarizing_error(0.01 * scale_factor, 1), ['h'])
    job = execute(qc, simulator, noise_model=scaled_model, shots=1024)
    results.append(job.result().get_counts())
print("Noisy Results:", results)
```

### 4. Zero-Noise Extrapolation
Fit noisy results to a polynomial model and extrapolate to zero noise.

```python
import numpy as np
from scipy.optimize import curve_fit

# Define the fit function (e.g., linear or polynomial)
def linear_fit(x, a, b):
    return a * x + b

# Fit noisy results to the model
noise_levels = [1.0, 1.5, 2.0]
noisy_data = [0.49, 0.42, 0.35]  # Example data for '00' counts
params, _ = curve_fit(linear_fit, noise_levels, noisy_data)

# Extrapolate to zero noise
zero_noise_value = linear_fit(0, *params)
print("Zero-Noise Estimate:", zero_noise_value)
```

---

## Results
- **Noise Impact**:
  - Amplified noise levels caused noticeable deviations in the circuit's output, highlighting the need for mitigation.
- **Extrapolation Success**:
  - The zero-noise estimate closely aligned with theoretical values, demonstrating ZNE's effectiveness.
- **Improved Accuracy**:
  - ZNE significantly reduced errors, improving computational reliability.

---

## How to Run the Project
1. **Install Dependencies**:
   ```bash
   pip install qiskit matplotlib numpy scipy
   ```
2. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
3. **Run the Jupyter Notebook**:
   - Open `simulation_with_zne.ipynb` in Jupyter Notebook.
   - Execute cells step by step to see the results.

---

## Future Work
- **Multiple Mitigation Techniques**:
  - Combine ZNE with methods like error cancellation, dynamical decoupling, and measurement error mitigation.
- **Refined Noise Models**:
  - Use device-specific noise models to improve accuracy.
- **Scaling to Complex Circuits**:
  - Apply ZNE to multi-qubit circuits and more complex quantum algorithms.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
- **Qiskit Team**: For developing the Qiskit framework.
- **IBM Quantum**: For providing open-source tools and simulators for quantum computing research.
