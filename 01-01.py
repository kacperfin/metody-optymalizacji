import cvxpy as cp

# x1: pszenica, x2: soja, x3: mączka
x = cp.Variable(3, nonneg=True)

# Minimalizacja kosztu: 300*x1 + 500*x2 + 800*x3
koszt = [300, 500, 800] @ x
cel = cp.Minimize(koszt)

ograniczenia = [
    [0.8, 0.3, 0.1] @ x >= 0.3, # Węglowodany
    [0.01, 0.4, 0.7] @ x >= 0.7, # Białko
    [0.15, 0.1, 0.2] @ x >= 0.1  # Sole
]

cp.Problem(cel, ograniczenia).solve()

print(f"Ilości składników: {x.value.round(4)}")
print(f"Koszt całkowity: {cel.value:.2f} zł")