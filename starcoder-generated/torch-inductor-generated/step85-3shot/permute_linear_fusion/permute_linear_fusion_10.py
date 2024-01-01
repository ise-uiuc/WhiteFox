
class Model(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        self.softmax = torch.nn.Softmax(-1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2 = v2.repeat(1, 1, 2)
        v3 = v2.detach()
        v3 = torch.sum(v3, dim=-1)
        return self.softmax(v3)
# Inputs to the model
x1 = torch.randn(1, 2, 4)
