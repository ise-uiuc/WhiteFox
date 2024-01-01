
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(19200, 4608)
        self.linear2 = torch.nn.Linear(4608, 9216)
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.linear2(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 19200)
