
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        w1 = torch.matmul(x1[0].unsqueeze(0), torch.eye(2))
        v1 = torch.nn.functional.linear(x1, w1, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        return v1.permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(2, 1, 2)
