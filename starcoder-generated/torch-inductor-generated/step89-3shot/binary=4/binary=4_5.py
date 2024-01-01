
class Model(torch.nn.Module):
    def __init__(self, n_output):
        super().__init__()
        self.linear = torch.nn.Linear(8, n_output)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return v1 + other

# Initializing the model
m = Model(2)

# Inputs to the model
x1 = torch.randn(64, 8)
