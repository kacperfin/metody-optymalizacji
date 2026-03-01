from pathlib import Path
import cvxpy as cp
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

script_dir = Path(__file__).parent
data = sio.loadmat(script_dir.parent / 'isoPerimData.mat')

N = int(data['N'][0, 0]) # liczba przedziałów
a = float(data['a'][0, 0]) # koniec przedziału (początek to 0)
L = float(data['L'][0, 0]) # maksymalna długość krzywej
C = float(data['C'][0, 0]) # maksymalna krzywizna
y_fixed = data['y_fixed'].flatten()

# Python indeksuje od 0, a MATLAB od 1. 
F = data['F'].flatten() - 1 

h = a / N # krok dyskretyzacji
x = np.linspace(0, a, N + 1) # punkty na osi X

y = cp.Variable(N + 1)

# pole to suma wartości * krok (h)
cel = cp.Maximize(h * cp.sum(y))

ograniczenia = [
    # warunki brzegowe (sznurek zaczyna się i kończy na osi X)
    y[0] == 0,
    y[N] == 0,
    
    # sznurek przechodzi przez wyznaczone paliki
    y[F] == y_fixed[F],
    
    # ograniczenie krzywizny
    cp.abs(cp.diff(y, k=2)) / (h**2) <= C,
    
    # ograniczenie długości krzywej
    cp.sum(cp.norm(cp.vstack([np.ones(N) * h, cp.diff(y)]), axis=0)) <= L,
]

problem = cp.Problem(cel, ograniczenia)
problem.solve()

print(f"Status rozwiązania: {problem.status}")
print(f"Maksymalne pole: {problem.value:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(x, y.value, 'k-', linewidth=2, label='Optymalna krzywa')
plt.plot(x[F], y.value[F], 'ko', markersize=8, label='Zadane punkty (F)')
plt.plot([0, a], [0, 0], 'ko', markersize=8) # Punkty początkowy i końcowy
plt.xlabel('x/a', fontsize=12)
plt.ylabel('y(x)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plots_dir = script_dir / 'wykresy'
plots_dir.mkdir(exist_ok=True)

plt.savefig(plots_dir / 'zadanie1-maksymalizacja.png')