
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.unsqueeze(1)
        x2 = v3
        x3 = x2.detach()
        x4 = torch.cat((x3, x2), -2)
        x4 *= 5
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
