
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)
 
    def forward(self, x1):
        k1 = 1.0
        v1 = self.linear(x1, k1)
        v2 = v1 + k1
        v3 = v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
output_1 = m(x1)

# Another input to the model
x2 = torch.randn(1, 1)
output_2 = m(x2)

