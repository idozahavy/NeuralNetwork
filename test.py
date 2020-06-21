from sympy import Symbol

from AILostCause.BetterNode import Node

x = Symbol("x")
y = x ** 2 + 5 * (x+4)

result = y.subs(x, 5)

print(type(result))
print(result)

print(type(y))
print(y)

print(y.diff("x"))

start_node = Node()
second_node = Node()
third_node = Node()
output_node = Node()

output_node.AddInputNode(third_node)
third_node.AddInputNode(second_node)
second_node.AddInputNode(start_node)

start_node.SetActivation(0.1)
output_node.CalcActivation()
print(f"Activation = {output_node.activation}")
formula = output_node.FormulateFactorNodeActivation(second_node.input_links[0])
print(f"first conn formula = {formula}")
diff_start = formula.diff("x")
print(f"diff formula = {diff_start}")
diff_start_5 = diff_start.subs(x, 0.1).evalf()
print(f"diff formula at point 0.1 = {diff_start_5}")
print()
print()

second_node.input_links[0].factor += (10 ** 14)

output_node.CalcActivation(recalculate=True)
print(f"Activation = {output_node.activation}")
formula = output_node.FormulateFactorNodeActivation(second_node.input_links[0])
print(f"first conn formula = {formula}")
diff_start = formula.diff("x")
print(f"diff formula = {diff_start}")
diff_start_5 = diff_start.subs(x, 0.1).evalf()
print(f"diff formula at point 0.1 = {diff_start_5}")
print()
print()
