
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convt = torch.nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.convt(x)
        return torch.sigmoid(v1)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16, 64, 64)
