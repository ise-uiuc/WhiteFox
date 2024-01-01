
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convt = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        x2 = self.convt(x)
        return F.relu(x2)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
