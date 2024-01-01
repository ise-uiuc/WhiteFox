
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(256, 256)
    def forward(self, x3):
        v1 = self.fc(x3)
        v2 = - v1
        v3 = v2 - 1
        v4 = v3 - 1.234
        v5 = v4 - 1.234e-05
        return - v4
# Inputs to the model
x3 = torch.randn(1, 256)
