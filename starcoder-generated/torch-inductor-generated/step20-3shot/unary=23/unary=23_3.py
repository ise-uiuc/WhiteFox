
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convTranspose = torch.nn.ConvTranspose2d(5, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.convTranspose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 5, 12, 12)
