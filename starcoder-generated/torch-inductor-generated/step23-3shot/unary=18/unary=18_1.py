
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sig = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(9216, 10)
    def forward(self, x1):
        v1 = self.sig(x1)
        v2 = v1.view(v1.size(0), -1)
        v3 = self.fc2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 2, 4, 12)
