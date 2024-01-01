
class Module0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter0 = torch.nn.Parameter(torch.randn([5, 5]))
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1)
        x3 = self.parameter0 + torch.rand_like(x2)
        return x3
# Inputs to the model
x1 = torch.randn(1, 5, 5)
