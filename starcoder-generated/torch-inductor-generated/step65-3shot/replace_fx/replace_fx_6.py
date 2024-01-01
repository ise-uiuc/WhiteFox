
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = torch.nn.Dropout(p=0.5)
    def forward(self, x1):
        x2 = self.mod(x1)
        x3 = torch.rand_like(x1)
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
