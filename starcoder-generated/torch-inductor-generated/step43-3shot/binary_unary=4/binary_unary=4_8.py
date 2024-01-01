, note use of custom class `Plus`
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = Plus(alpha=0.5)(v1)
        v3 = self.linear(v2)
        v4 = Plus(alpha=0.7071067811865476)(v1)
        v5 = relu(v3)
        v6 = Plus(alpha=0.5)(v4)
        w = Plus()(v3, v6)
        return v5, w

class Plus(torch.nn.Module):
    def __init__(self, alpha=1, beta=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
 
    def forward(self, x1, x2):
        y = self.alpha * x1 + self.beta * x2
        return y
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(2, 3)
