
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(64 * 64, 100*100)
 
    def forward(self, x1):
        w1 = self.linear(x1)
        w2 = torch.clamp_min(w1, min_value)
        w3 = torch.clamp_max(w2, max_value)
        return w3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64 * 64)
