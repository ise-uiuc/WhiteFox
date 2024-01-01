
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = self.softmax(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 5, 4)
