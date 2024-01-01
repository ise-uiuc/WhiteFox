
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(192, 2)
    def forward(self, x1, x2, x3, x4):
        v1 = torch.cat((x1, x2, x3, x4), 1)
        v2 = v1.size()[1]
        v3 = self.fc1(v1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 96, 4, 4)
x2 = torch.randn(1, 32, 14, 14)
x3 = torch.randn(1, 64, 7, 7)
x4 = torch.randn(1, 128, 7, 7)
