
class Model(torch.nn.Module):
    def __init__(self, hidden_size=10):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * (v1.clamp(min=0, max=6) + 3)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
