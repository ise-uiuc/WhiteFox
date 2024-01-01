
class Model(torch.nn.Module):
    def _init_(self):
        super().__init__()
 
    def forward(self, X0, X1):
        v6 = X0 * 0.25 # Mulitply the input by 0.25
        v5 = X0 * 0.3673 # Mulitply the input by 0.3673
        v7 = v6 + X1 # Add the two inputs
        v8 = v5 - X1 # Substract the two inputs
        return v7, v8

# Initializing the model and an input
# Inputs to the model
x1 = torch.randn(16, 28, 4, 10)
x2 = torch.randn(1, 10)
__a__, __b__ = m(x1, x2)

