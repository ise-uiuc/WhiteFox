
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.bn = torch.nn.BatchNorm2d(1, affine=False)
        self.relu = torch.nn.ReLU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=-1)
    def forward(self, x1):
        v0 = x1
        v1 = v0.permute(0,2,1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.bn(v2.unsqueeze(1)).squeeze(1)
        v4 = self.relu(v3)
        v5 = self.softmax(v4)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
