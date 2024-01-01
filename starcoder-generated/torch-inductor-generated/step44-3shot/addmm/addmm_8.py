
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 3)
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, inp)
        v2 = self.fc(x2)
        return self.fc(v1) + v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
