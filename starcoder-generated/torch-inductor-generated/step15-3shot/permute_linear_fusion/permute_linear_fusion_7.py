
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(576, 224)
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1) + x2
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.T.unsqueeze(-3)
        x3 = v3 + v1
        x3.permute(0, 2, 1).T.transpose(-2, -1) + v2.unsqueeze(-3) + v2
        return x3 + v3
# Inputs to the model
x1 = torch.randn(1, 224, 298)
x2 = torch.randn(1, 32, 224)
