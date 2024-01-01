
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 1)
    def forward(self, x1):
        v3 = torch.sum(self.linear1(x1), dim=1)
        x2 = torch.tanh(v3)
        v1 = x2 * x2
        v2 = torch.squeeze(self.linear2(v1), dim=1)
        v4 = torch.abs(v2)
        return v4
# Inputs to the model
x1 = torch.randn(2, 2)
