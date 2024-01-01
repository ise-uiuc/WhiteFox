
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2).to(torch.float16)
    def forward(self, x1):
        v2 = x1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v2, self.linear.weight.permute(0, 2, 1) @ self.linear.weight, self.linear.bias)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
