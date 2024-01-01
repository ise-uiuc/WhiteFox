
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.hardtanh = torch.nn.Hardtanh(min_val=-1, max_val=1)
        self.unsqueeze = torch.Tensor.unsqueeze
        self.eq = torch.Tensor.__eq__
        self.sum = torch.Tensor.sum
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x1):
        v1 = self.relu(x1)
        v1 = self.hardtanh(v1)
        v2 = self.unsqueeze(v1, -1)
        v3 = self.eq(v2, -1)
        v3 = v3.to(v1.dtype)
        v3 = self.linear(v3)
        v2 = v1.permute(1, 0)
        v2 = v2[0:1, 0:1]
        v3 = self.sum(v3.to(v1.dtype))
        return v2
# Inputs to the model
x1 = torch.randn(2, 1)
