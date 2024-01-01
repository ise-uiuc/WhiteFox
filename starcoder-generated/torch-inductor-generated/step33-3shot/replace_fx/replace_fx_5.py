
class model(torch.nn.Module):
    def __init__(self, x):
        super().__init__()
        self.parameter0 = torch.nn.Parameter(torch.randn([5, 5]))
    def forward(self, x1):
        x2 = torch.rand_like(x1, dtype=torch.float)
        return x2
# Inputs to the model
x1 = torch.randn(3, 5, 5)
