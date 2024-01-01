
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        v3 = torch.nn.functional.hardtanh(v2)
        v1 = x1.permute(0, 2, 1)
        v4 = torch.nn.functional.hardtanh(v2.permute(0, 2, 1) - x1)
        v2 = torch.nn.functional.hardtanh(v2 - x1.permute(0, 2, 1))
        v2 = torch.nn.functional.hardtanh(-v2)
        v5 = torch.nn.functional.hardtanh(torch.nn.functional.hardtanh(v2 - x1.permute(0, 2, 1)))
        v1 = x1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        v2 = x1.permute(0, 2, 1)
        v4 = torch.nn.functional.linear(v3, self.linear1.weight, self.linear1.bias)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
