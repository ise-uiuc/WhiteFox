
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(120)
        self.flatten1 = torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.dropout1 = torch.nn.Dropout()
        self.linear1 = torch.nn.Linear(144*144, 120)
        self.linear2 = torch.nn.Linear(120, 20)
    def forward(self, x1):
        v1 = self.bn1(x1)
        v2 = self.flatten1(v1)
        v3 = self.dropout1(v2)
        v4 = self.linear1(v3)
        v5 = self.linear2(v4)
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 120, 144, 144)
