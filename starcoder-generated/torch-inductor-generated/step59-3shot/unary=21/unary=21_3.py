
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(3, 4)
        self.linear3 = torch.nn.Linear(4, 3)
        self.linear4 = torch.nn.Linear(3, 2)
        self.linear5 = torch.nn.Linear(2, 2)
        self.linear6 = torch.nn.Linear(2, 2)
    def forward(self, x4):
        x5 = self.linear1(x4)
        x6 = torch.tanh(x5)
        x7 = self.linear2(x6)
        x8 = torch.tanh(x7)
        x9 = self.linear3(x8)
        x10 = torch.tanh(x9)
        x11 = self.linear4(x10)
        x12 = torch.tanh(x11)
        x13 = self.linear5(x12)
        x14 = torch.tanh(x13)
        x15 = self.linear6(x14)
        return x15
# Inputs to the model
x4 = torch.randn(6, 2)
