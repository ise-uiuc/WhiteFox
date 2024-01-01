
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 32, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv__t(x)
        v2 = sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
