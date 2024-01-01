
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_0 = torch.nn.Linear(in_features=60000, out_features=30, bias=False)
        self.lin_1 = torch.nn.Linear(in_features=30, out_features=80, bias=False)
        self.lin_2 = torch.nn.Linear(in_features=30, out_features=80, bias=False)
        self.lin_3 = torch.nn.Linear(in_features=80, out_features=50, bias=False)
        self.lin_4 = torch.nn.Linear(in_features=50, out_features=10, bias=False)
    def forward(self, x1, x2):
        x1 = self.lin_0(x1)
        x1 = self.lin_1(x1)
        x1 = self.lin_2(x1)
        x1 = self.lin_3(x1)
        x1 = self.lin_4(x1)
        return torch.cat([x1, x1], 1)
# Inputs to the model
x1 = torch.randn(1, 60000)
x2 = torch.randn(1, 30)
