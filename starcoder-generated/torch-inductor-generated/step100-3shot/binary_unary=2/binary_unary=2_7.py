
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv1d(16, 33, 3)
        self.c2 = torch.nn.Conv1d(33, 33, 3)
    def forward(self, x1):
        v1 = self.c1(x1)
        v2 = v1 - 1.1
        v3 = F.relu(v2)
        v4 = self.c2(v3)
        v5 = v4 - 1.1
        v6 = F.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 16, 232)
