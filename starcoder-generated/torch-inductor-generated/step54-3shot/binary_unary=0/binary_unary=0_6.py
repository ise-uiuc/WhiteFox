
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 1, bias=False)
        self.linear2 = torch.nn.Linear(1, 1, bias=False)
    def forward(self, x1, x2):
        v1 = torch.squeeze(self.linear1(x1))
        v2 = torch.squeeze(self.linear2(x2))
        v3 = v1 + v2
        v4 = torch.relu(v3)
        v5 = torch.squeeze(self.linear1(x2))
        v6 = torch.squeeze(self.linear2(x2))
        v7 = v5 + v6
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 1)
x2 = torch.randn(1, 1, 1)
