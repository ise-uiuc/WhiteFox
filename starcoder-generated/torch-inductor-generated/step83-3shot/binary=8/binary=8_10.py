
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3820, 2048)
        self.linear2 = torch.nn.Linear(2048, 1000)
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = v1.transpose(-1, -2).add(x)
        v3 = self.linear2(v2)
        return v3
# Inputs to the model
x = torch.randn(256, 1000)
