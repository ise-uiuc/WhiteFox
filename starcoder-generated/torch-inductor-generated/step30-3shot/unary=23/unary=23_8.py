
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.add = torch.nn.ConvTranspose2d(1, 2, stride=3)
    def forward(self, x1):
        v1 = self.add(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(14, 1, 72, 6) # Can you guess why would this one fail?
