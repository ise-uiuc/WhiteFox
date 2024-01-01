
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 16)
    def forward(self, x1, x2):
        v1 = x2.permute(0, 2, 1)
        v1 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        # PyTorch API'match_x'
        v2 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        x2 = v1 + v2
        x2 = x2.permute(0, 2, 1)
        x2 = torch.sigmoid(x2 + v2)
        v2 = torch.min(x2, dim=-1)[1]
        x2 = v1 * x2
        x3 = x2.permute(0, 2, 1)
        x3 = x3 * v1
        return x3
# Inputs to the model
x1 = torch.randn(1, 16, 8)
x2 = torch.randn(1, 32, 4)
