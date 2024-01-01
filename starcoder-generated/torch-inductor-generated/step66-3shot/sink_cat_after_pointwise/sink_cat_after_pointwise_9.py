
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(3, 5)
        self.layer2 = torch.nn.Linear(5, 7)
    def forward(self, x):
        return torch.cat([self.layer1(x), self.layer2(x)], dim=1).tanh
# Inputs to the model
x = torch.randn(2, 3, 4)
