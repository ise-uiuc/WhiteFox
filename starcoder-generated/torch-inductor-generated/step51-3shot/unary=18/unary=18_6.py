
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features=1152, out_features=1000)
        self.linear2 = torch.nn.Linear(in_features=1000, out_features=2)
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = torch.tanh(v1)
        v3 = self.linear2(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1152)
