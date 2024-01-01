
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.softmax = torch.nn.Softmax(dim=-1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.matmul(v2, self.linear.bias)
        y = torch.matmul(v1, self.linear.weight)
        y1 = torch.matmul(y, x2.transpose(1, 2))
        y2 = self.softmax(y)
        z = torch.bmm(y2, y1)
        x3 = z.transpose(1, 2)
        return x3
# Inputs to the model
x1 = torch.randn(1, 3, 2)
