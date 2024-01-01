
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 2)
    def forward(self, x, mask):
        z = self.layer1(x)
        return z * mask
# Inputs to the model
x = torch.randn(1, 2, 2)
mask = torch.ones([1, 1, 2])
