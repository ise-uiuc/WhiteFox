
class Model(torch.nn.Module):
    def __init__(self, max_size=200):
        super().__init__()
        self.fc = torch.nn.Linear(21, 50)
        self.max_size = max_size
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = torch.clamp_max(v1, self.max_size)
        return v2
# Inputs to the model
x1 = torch.randn(1, 21)
