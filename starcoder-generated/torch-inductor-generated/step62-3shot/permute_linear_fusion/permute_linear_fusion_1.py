
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.flatten1 = torch.nn.Flatten()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v1 = torch.nn.functional.relu(v1)
        v1 = self.linear(v1)
        v1 = torch.nn.functional.softmax(v1, 1)
        v1 = torch.max(v1, dim=-1)[0]
        v2 = self.flatten1(v1)
        x2 = torch.matmul(v1, self.linear.bias)
        x2 = torch.mean(x2, dim=-1) + v2
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
