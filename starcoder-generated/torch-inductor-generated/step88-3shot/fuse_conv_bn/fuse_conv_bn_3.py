
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense0 = torch.nn.Linear(4, 32, bias=False)
        self.dense1 = torch.nn.Linear(32, 32, bias=False)
        self.dense2 = torch.nn.Linear(32, 2, bias=False)
        self.bn2 = torch.nn.BatchNorm1d(32, affine=False)
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.dense0(x)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.sigmoid(x)
        return x
# Inputs to the model
x = torch.randn(2, 2, 2, 2)
