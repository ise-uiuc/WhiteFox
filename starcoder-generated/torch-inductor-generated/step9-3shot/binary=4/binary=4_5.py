 with a residual connection
class ModelWithResidualConnection(torch.nn.Module):
    def __init__(self, a=10.0):
        super().__init__()
        self.other = torch.nn.Parameter(torch.randn(1, 16, 64, 64))
        self.linear = torch.nn.Linear(16*64*64, 100)
 
    def forward(self, x1):
        v1 = self.linear(x1.view(1, 16*64*64))
        v2 = v1 + self.other.view(1, 16*64*64)
        return v2
 
m = ModelWithResidualConnection()
x1 = torch.randn(1, 16, 64, 64)
