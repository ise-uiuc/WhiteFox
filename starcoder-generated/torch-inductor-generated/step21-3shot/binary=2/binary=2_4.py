
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(224, 1000)
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 - 0.747
        return v2
# Inputs to the model
x1 = torch.randn(1, 224)
