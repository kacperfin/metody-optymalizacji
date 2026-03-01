import cvxpy as cp

# x1: płatki, x2: mleko, x3: chleb
x = cp.Variable(3, nonneg=True)

# Minimalizacja kosztu: 0.15*x1 + 0.25*x2 + 0.05*x3
koszt = [0.15, 0.25, 0.05] @ x
cel = cp.Minimize(koszt)

witaminy = [107, 500, 0] @ x
cukier = [45, 40, 60] @ x
kalorie = [70, 121, 65] @ x

ograniczenia = [
    kalorie >= 2000,
    kalorie <= 2250,
    witaminy >= 5000,
    witaminy <= 10000,
    cukier <= 1000,
    x <= 10 # Maksymalnie 10 porcji każdego typu
]

cp.Problem(cel, ograniczenia).solve()

print(f"Ilość porcji (płatki, mleko, chleb): {x.value.round(4)}")
print(f"Koszt optymalny: {cel.value:.4f}")
