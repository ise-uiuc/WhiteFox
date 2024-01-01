
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convt = torch.nn.ConvTranspose2d(8, 4, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.convt(x)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8, 64, 64)
