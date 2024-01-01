
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.softmax = torch.nn.Softmax(-1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.gelu(v2)
        x3 = x2.squeeze(dim=2)
        v4 = self.softmax(x3)
        v5 = v4.unsqueeze(dim=2)
        x6 = torch.matmul(v5, x2)
        return x6
# Inputs to the model
x1 = torch.randn(1, 2, 3)
