import pulp

lp = pulp.LpProblem('MyModel', pulp.LpMinimize)

variables = [pulp.LpVariable(f'x{i}{k}', lowBound=0) for i in range(1, 4) for k in range(1, 6)]
x11, x12, x13, x14, x15, x21, x22, x23, x24, x25, x31, x32, x33, x34, x35 = variables

lp += x11 + x21 + x31 == 700 	
lp += x12 + x22 + x32 == 300 	
lp += x13 + x23 + x33 == 900 	
lp += x14 + x24 + x34 == 600 	
lp += x15 + x25 + x35 == 500 

lp += x11 + x12 + x13 + x14 + x15 <= 1200	
lp += x21 + x22 + x23 + x24 + x25 <= 1200	
lp += x31 + x32 + x33 + x34 + x35 <= 1200	

lp += 8 * x11 + 12 * x12 + 9 * x13 + 8 * x14 + 0 * x15 + 11 * x21 + 9 * x22 + 16 * x23 + 0 * x24 + 8 * x25 + 14 * x31 + 0 * x32 + 10 * x33 + 9 * x34 + 12 * x35

lp.solve(pulp.PULP_CBC_CMD(msg=False))

print(lp)
print('-' * 50)
for variable in variables:
    print(f'{variable}:  {variable.value()}')
objective_result = pulp.value(lp.objective)
print(f'Amaç fonksiyonun değer: {objective_result }')

