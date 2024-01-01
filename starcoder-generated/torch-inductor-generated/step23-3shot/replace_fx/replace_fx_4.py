
class m1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=in_features=2, out_features=2)
        self.bn = torch.nn.BatchNorm1d(num_features=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        x2 = self.bn(x1)
        x3 = x2 ** 2
        x4 = torch.nn.functional.dropout(x3)
        x5 = torch.randn(1, 1, 2)
        x6 = self.linear(x2) + x4
        return x6
in_features = 2
num_samples = 1
# Inputs to the model
x1 = torch.randn(num_samples, in_features)
