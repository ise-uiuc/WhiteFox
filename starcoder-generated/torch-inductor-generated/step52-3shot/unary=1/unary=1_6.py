
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 10, True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * 0.5
        # Add the output of the linear transformation to the output of the linear transformation cubed multiplied by 0.044715
        v3 = v1 + (torch.pow(torch.abs(v1), 3)) * (0.3534238441943475)
        v4 = v3 * 0.7978845608028654
        v5 = torch.tanh(v4)
        v6 = v5 + 1
        v7 = v2 * v6
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
