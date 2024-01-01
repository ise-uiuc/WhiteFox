
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(in_features=64, out_features=128)
 
    def forward(self, x1, __min__, __max___):
        v1 = self.fc(x1)
        v2 = torch.clamp_min(v1, __min__)
        v3 = torch.clamp_max(v2, __max___)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
m_min = -1
m_max = 1
