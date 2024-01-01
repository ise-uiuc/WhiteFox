
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 1)
        self.out = torch.nn.Linear(1, 1)
    def forward(self, x1, x2, x3, x4):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x3, x4)
        v3 = v1 + v2
        v4 = self.fc1(v3)
        return v4
#Inputs to the model
x1 = torch.randn(1, 65)
x2 = torch.randn(65, 5)
x3 = torch.randn(1, 65)
x4 = torch.randn(65, 5)
#Model ends
