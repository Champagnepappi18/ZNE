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
from qiskit import QuantumCircuit

# Define the quantum circuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
```

### 2. Noise Model Configuration
Simulate realistic noise using depolarizing noise models.

```python
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel, depolarizing_error

# Define noise model
noise_model = NoiseModel()
# Add 1-qubit depolarizing error
one_qubit_error = depolarizing_error(0.01, 1)
# Add 2-qubit depolarizing error for cx gate
two_qubit_error = depolarizing_error(0.02, 2)
# Apply errors to specific gates
noise_model.add_all_qubit_quantum_error(one_qubit_error, ['h'])
noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])
```

### 3. Folded Circuit and Noise Amplification
Create folded circuits and scale noise levels for data collection.

```python
# Create a folded circuit (excluding the measurement)
def create_folded_circuit(original_qc, scale_factor):
    gates_only_qc = original_qc.copy()
    gates_only_qc.data = gates_only_qc.data[:-2]  # Remove measurement operations
    folded_qc = gates_only_qc
    for _ in range(int(scale_factor - 1)):
        folded_qc = folded_qc.compose(gates_only_qc.inverse())
    folded_qc.measure_all()
    return folded_qc

# Run the circuit at different noise levels
def run_with_noise(qc, scale_factor, noise_model):
    folded_qc = create_folded_circuit(qc, scale_factor)
    backend = Aer.get_backend('qasm_simulator')
    job = backend.run(folded_qc, noise_model=noise_model, shots=1024)
    result = job.result()
    return result.get_counts()

# Collect noisy results
scale_factors = [1, 1.5, 2]
results = [run_with_noise(qc, sf, noise_model) for sf in scale_factors]

print("Noisy Counts:", results)  # Debugging counts
```

### 4. Zero-Noise Extrapolation
Fit noisy results to a polynomial model and extrapolate to zero noise.

```python
import numpy as np
from scipy.optimize import curve_fit

# Fit linear model
def linear_model(x, a, b):
    return a + b * x

counts = np.array([result.get('00', 0) for result in results])  # Adjust for 2-qubit counts
params, _ = curve_fit(linear_model, scale_factors, counts)
noiseless_result = params[0]

# Quantitative analysis
true_result = 512  # Example true result
noisy_result = counts[0]
zero_noise_error = abs(noiseless_result - true_result)
noisy_error = abs(noisy_result - true_result)
error_reduction = ((noisy_error - zero_noise_error) / noisy_error) * 100

print(f"True Result: {true_result}")
print(f"Noisy Result: {noisy_result}")
print(f"Noisy Error: {noisy_error}")
print(f"Zero-Noise Error: {zero_noise_error}")
print(f"Error Reduction: {error_reduction:.2f}%")
```

### 5. Visualization
Plot extrapolation and error metrics for analysis.

```python
import matplotlib.pyplot as plt

# Plot extrapolation results
plt.figure(figsize=(10, 6))
plt.scatter(scale_factors, counts, label="Noisy Results", color="blue")
plt.plot(scale_factors, linear_model(np.array(scale_factors), *params), label="Linear Fit", color="red")
plt.axhline(y=noiseless_result, color="green", linestyle="--", label="Zero-Noise Estimate")
plt.axhline(y=true_result, color="purple", linestyle=":", label="True Result")
plt.xlabel("Scale Factor")
plt.ylabel("Counts for '00'")
plt.title("Extrapolation to Zero Noise")
plt.legend()
plt.grid(True)
plt.show()

# Plot errors
metrics = ["Noisy Error", "Zero-Noise Error", "Error Reduction (%)"]
values = [noisy_error, zero_noise_error, error_reduction]

plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color=["blue", "green", "orange"])
plt.title("Error Metrics Comparison")
plt.ylabel("Values")
plt.grid(axis="y", linestyle="--")
plt.show()
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
