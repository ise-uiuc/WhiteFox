
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ct = torch.nn.ConvTranspose1d(55, 25, 6, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.ct(x1)
        v2 = torch.sigmoid(v1)
        return v2    
# Inputs to the model
x1 = torch.randn(12, 55, 256)
