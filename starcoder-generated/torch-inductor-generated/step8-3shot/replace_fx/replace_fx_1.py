
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 10, 3)
    def forward(self, x1):
        a1 = self.conv(x1)
        a2 = torch.nn.functional.dropout(a1)
        a3 = torch.rand_like(a1, dtype=torch.float)
        a4 = a3 - torch.randn(1)
        return a2 / a4
# Inputs to the model
x1 = torch.randn(1, 10, 50, 50)
