
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 20)
    def forward(self, x1):
        v1 = x1.transpose(-1, -2)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2 = v2.reshape(2, 2, 10)
        v3 = v2.transpose(2, 1)
        v4 = torch.mean(v3)
        v5 = v4.unsqueeze_(-1).unsqueeze(-1)
        v6 = v5.expand_as(v3)
        x2 = v3 * v6
        return x2
# Inputs to the model
x1 = torch.randn(1, 20)
