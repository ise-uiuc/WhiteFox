
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(24, 32, bias=False)
        weight = torch.randn(32, 24, dtype=torch.float32)
        self.linear.weight.data = weight.data/weight.abs().max()
        self.linear.weight.requires_grad_(False) # We don't need grad for the new weight
        self.other = torch.randn(24, 1, dtype=torch.float)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        v3 = torch.relu(v2)
        return v3

