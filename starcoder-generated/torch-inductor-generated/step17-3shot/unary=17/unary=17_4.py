
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pdc = torch.nn.ConvTranspose2d(3, 32, 5, padding=2, stride=2)
    def forward(self, x1):
        v1 = self.pdc(x1)
        v2 = F.relu(v1)
        return F.tanh(v2)
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
