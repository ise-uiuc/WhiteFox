
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(128,32)
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        return torch.cat([v1, v1, v1, v1, v1], 1)
# Inputs to the model
x1 = torch.randn(128,32)
x2 = torch.randn(32,128)
