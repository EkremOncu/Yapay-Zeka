import pulp

lp = pulp.LpProblem('MyModel', pulp.LpMaximize)

x1 = pulp.LpVariable(name='x1', lowBound=0)
x2 = pulp.LpVariable(name='x2', lowBound=0)

lp += 6 * x1 + 4 * x2 <= 120
lp += 3 * x1 + 10 * x2 <= 180
lp += 45 * x1 + 55 * x2

lp.solve(pulp.PULP_CBC_CMD(msg=False))

print(lp)
print('-' * 50)

x1_result = x1.value()
x2_result = x2.value()

print(f'x1 = {x1_result}, x2 = {x2_result}')
objective_result = pulp.value(lp.objective)
print(f'Amaç fonksiyonun değer: {objective_result }')

