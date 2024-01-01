
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.hardtanh = torch.nn.Hardtanh(min_val=0, max_val=5)
        self.hardswish = torch.nn.Hardswish()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        return self.hardswish(self.hardtanh(v2))
# Inputs to the model
x1 = torch.randn(1, 2, 2)
