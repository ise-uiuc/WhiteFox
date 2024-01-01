
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight)
        v3 = torch.nn.Sigmoid()(v2)
        v3 = self.linear(v3)
        v4 = self.linear(v3)
        x2 = torch.matmul(v4, self.linear.weight)
        z = (x1 * 2) ** v3
        return x2 + z
# Inputs to the model
x1 = torch.randn(1, 4, 4)
