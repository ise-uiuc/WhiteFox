
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
    def forward(self, x3):
        v3 = torch.nn.functional.linear(x3, self.linear.weight, self.linear.bias)
        v4 = torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias)
        v1 = v4.permute(0, 2, 1)
        v2 = v1.contiguous()
        return v2
# Inputs to the model
x3 = torch.randn(1, 2, 2)
