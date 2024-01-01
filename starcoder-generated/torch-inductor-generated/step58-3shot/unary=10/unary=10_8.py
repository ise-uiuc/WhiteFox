
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 64)
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = v1 + 3
        v3 = F.relu6(v2, inplace=True)
        v4 = F.hardtanh(v3, min_val=0.0, max_val=6.0)
        v5 = v4 / 6
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
