
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=0)
        v3 = torch.clamp_max(v2, max_value=5)
        return v3

# Initializing the model
min_value = torch.randn(1).unsqueeze(dim=0).item()
max_value = torch.randn(1).unsqueeze(dim=0).item()
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
