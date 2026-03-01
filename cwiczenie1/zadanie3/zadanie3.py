import cvxpy as cp

# zmienne:
# x[0]: Lek I (1000 opakowań)
# x[1]: Lek II (1000 opakowań)
# x[2]: Surowiec I (kg)
# x[3]: Surowiec II (kg)
x = cp.Variable(4, nonneg=True)

# zmienne: Lek I, Lek II, Surowiec I, Surowiec II
cena_sprzedazy = [6500, 7100, 0, 0] @ x
zawartosc_A = [-0.5, -0.6, 0.01, 0.02] @ x
zasoby_ludzkie = [0.5, 0.6, 0, 0] @ x
zasoby_sprzetowe = [40, 50, 0, 0] @ x
koszty_operacyjne = [700, 800, 0, 0] @ x
cena_zakupu_surowca = [0, 0, 100.00, 199.90] @ x

f_koszt = cena_zakupu_surowca + koszty_operacyjne
f_przychod = cena_sprzedazy

cel = cp.Minimize(f_koszt - f_przychod)

ograniczenia = [
    f_koszt <= 100000,
    zasoby_ludzkie <= 2000,
    zasoby_sprzetowe <= 800,
    x[2] + x[3] <= 1000,
    zawartosc_A >= 0 # musi być więcej czynnika A niż jest potrzebne do wyprodukowania leków
]

problem = cp.Problem(cel, ograniczenia)
problem.solve()

print(f"Status problemu: {problem.status}")
print(f"Wartość optymalna: {problem.value:.2f} USD")
print(f"Lek I: {x[0].value:.3f} (1000 opakowań)")
print(f"Lek II: {x[1].value:.3f} (1000 opakowań)")
print(f"Surowiec I: {x[2].value:.3f} kg")
print(f"Surowiec II: {x[3].value:.3f} kg")
