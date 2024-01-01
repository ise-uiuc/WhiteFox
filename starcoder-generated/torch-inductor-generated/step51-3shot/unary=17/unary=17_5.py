
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 5)
    def forward(self, x1):
        v1 = self.fc(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3)
