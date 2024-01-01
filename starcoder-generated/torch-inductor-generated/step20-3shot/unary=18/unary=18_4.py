
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features=4, out_features=8)
        self.linear2 = torch.nn.Linear(in_features=8, out_features=4)
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = self.linear2(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 4)
