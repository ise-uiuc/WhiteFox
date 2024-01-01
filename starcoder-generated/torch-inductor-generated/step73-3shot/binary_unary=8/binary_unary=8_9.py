
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.linear = nn.Linear(1, 120)
        # self.linear = nn.Linear(120, 84)
        # self.linear = nn.Linear(84, 50)
        # self.linear = nn.Linear(50, 320)
        self.linear = torch.nn.Linear(320, 160)
        self.linear2 = torch.nn.Linear(160, 80)
        self.linear3 = torch.nn.Linear(80, 40)
        self.linear4 = torch.nn.Linear(40, 20)
        self.linear5 = torch.nn.Linear(20, 10)
    def forward(self, x):
        v1 = self.linear(x)
        v2 = self.linear2(v1)
        v3 = self.linear3(v2)
        v4 = self.linear4(v3)
        v5 = self.linear5(v4)
        return v5
# Inputs to the model
x = torch.randn(2, 1, 300, 450)
