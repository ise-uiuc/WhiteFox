
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 16)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + x1
        v3 = torch.relu(v2)
        return v3
x1 = torch.randn(1, 32)
x2 = torch.randn(1, 32)
