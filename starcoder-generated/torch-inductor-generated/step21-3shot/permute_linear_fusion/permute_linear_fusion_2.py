
class Model(torch.nn.Module):
    def __init__(self, out, in_features, bias=False):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out, bias=bias)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        return self.linear(v1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
