
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(64, 128)
 
    def forward(self, x1, min_value=0, max_value=0.5):
        v1 = torch.clamp_max(self.dense(x1), min_value=min_value)
        return torch.clamp_max(v1, max_value=max_value)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
