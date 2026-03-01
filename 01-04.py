import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog

data = pd.read_csv('data/01-04.csv', header=None)
x = data[0].values
y = data[1].values
N = len(x)

Phi = np.column_stack((x, np.ones(N)))

# Suma kwadratów różnic (LS)

theta_LS = np.linalg.pinv(Phi) @ y
a_LS, b_LS = theta_LS
print(f"Suma kwadratów różnic (LS): a = {a_LS:.4f}, b = {b_LS:.4f}")

# Suma wartości bezwzględnych różnic (LP)

c = np.zeros(N + 2)
c[2:] = 1

A_ub = np.vstack([
    np.hstack([Phi, -np.eye(N)]),
    np.hstack([-Phi, -np.eye(N)])
])
b_ub = np.hstack([y, -y])

res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None)] + [(0, None)] * N)

if res.success:
    theta_LP = res.x[:2]
    a_LP, b_LP = theta_LP
    print(f"Suma wartości bezwzględnych różnic (LP): a = {a_LP:.4f}, b = {b_LP:.4f}")
else:
    print("Niepowodzenie LP: ", res.message)

# Wizualizacja

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', s=10, label='Punkty')

y_LS = a_LS * x + b_LS
plt.plot(x, y_LS, color='black', linewidth=2, label=f'LS: y = {a_LS:.2f}x + {b_LS:.2f}')

y_LP = a_LP * x + b_LP
plt.plot(x, y_LP, color='blue', linewidth=2, label=f'LP: y = {a_LP:.2f}x + {b_LP:.2f}')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.ylim(-50, 150)
plt.savefig('plots/01-04-001.png')
plt.ylim(-10, 15)
plt.savefig('plots/01-04-002.png')