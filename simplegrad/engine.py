class Value:
    '''stores a single scalar value and its gradien along with activation functions'''
    #constructor 
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables, ie private variables
        self._backward = lambda : None
        self._prev = set(_children) # stores previous Value objects
        self._op = _op 

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self,other), '+' )

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other,(int, float)) # Throws AssertionError if the isinstance returns false
        out = Value(self.data ** other,(self,), f'**{other}' )

        def _backward():
            self.grad += (other * self.data ** (other -1)) * out.grad
        out._backward = _backward
        return out
    
    def __neg__(self): # -self
        return self * -1
    def __radd__(self,other): # other + self
        return self + other
    def __sub__(self,other): # self - other
        return self + (-other)
    def __rsub__(self, other): # other - self
        return other + (-self)
    def __rmul__(self,other): # other * self
        return self * other 
    def __truediv__(self, other): # self / other
        return self * (other * -1)
    def __rtruediv__(self,other): # other / self
        return other * self ** -1
    def __repr__(self): # output when printed the object
        return f"Value(data={self.data},grad = {self.grad})"

    # Activaiton functions
    def relu(self):
        out = Value(max(0,self.data), (self,),'relu')
        def _backward():
            self.grad += (out.data > 0 ) * out.grad
        return out
    
    # Topological graph
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1 # setting initial grad value to  1
        for v in reversed(topo):
            v._backward()
