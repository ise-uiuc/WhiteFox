
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 1)
    def forward(self, x1):
        t1 = x1**2
        t4 = self.fc(t1)
        return t4
# Inputs to the model
x1 = torch.randn(1, 4)
