
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dconv = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1, output_padding=0)
 
    def forward(self, x):
        v1 = self.dconv(x)
        v2 = torch.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
