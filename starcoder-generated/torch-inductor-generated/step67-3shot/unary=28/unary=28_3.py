
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.269059, max_value=0.315838):
        super().__init__()
        self.fc = torch.nn.Linear(3, 4)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = torch.clamp_max(v1, max=max_value)
        v3 = torch.clamp_min(v2, min=min_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
