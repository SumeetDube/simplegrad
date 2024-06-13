from simplegrad.engine import Value

a = Value(-2.0)
b = Value(3.0)
d = a * b    
e = a + b    
f = d * e    
f.backward()
print(f)
print(a)
print(b)
