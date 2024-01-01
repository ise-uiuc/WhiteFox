
class Model(torch.nn.Module):
    def __init__(self, min_value=0.6, max_value=0.9):
        super().__init__()
        self.linear = torch.nn.Linear(128, 64)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.clamp_min(v1, min_value=0.6)
        v3 = torch.clamp_max(v2, max_value=0.9)
        return v3

# Initializing the model
m = Model()
