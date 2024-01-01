
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
        self.t3 = torch.tensor(float("-inf")).to(torch.float32)
        self.t2 = torch.tensor(float("inf")).to(torch.float32)
 
    def forward(self, x1):
        v1 = self.linear(x1).clamp_min(float("-inf")).clamp_max(float("inf"))
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
min_value = -0.0001
max_value = 0.0001
