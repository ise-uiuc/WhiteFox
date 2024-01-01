
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1.permute(0, 2, 1)
        v3 = self.sigmoid(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 2, 2)
