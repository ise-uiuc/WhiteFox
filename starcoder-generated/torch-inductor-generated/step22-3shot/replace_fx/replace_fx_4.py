
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 4)
        self.linear2 = torch.nn.Linear(4, 4)
        self.linear3 = torch.nn.Linear(4, 4)
        self.linear4 = torch.nn.Linear(4, 4)
        self.linear5 = torch.nn.Linear(4, 4)
        self.linear6 = torch.nn.Linear(4, 4)
        self.linear7 = torch.nn.Linear(4, 4)
        self.linear8 = torch.nn.Linear(4, 4)
        self.linear9 = torch.nn.Linear(4, 4)
        self.linear10 = torch.nn.Linear(4, 4)
    def forward(self, x1):
        x2 = self.linear1(x1)
        x3 = self.linear2(x2)
        x4 = self.linear3(x3)
        x5 = self.linear4(x3)
        x6 = self.linear5(x4)
        x7 = self.linear8(x6)
        x8 = self.linear7(x7)
        x9 = self.linear9(x7)
        x10 = self.linear10(x9)
        x11 = x2 - x4
        return x11
# Inputs to the model
x1 = torch.randn(1, 1)
