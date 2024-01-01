
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 12, 3, stride=1, padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(12, 3, 3, stride=2, padding=1, output_padding=1)
 
    def forward(self, x):
        x = self.conv1(x)
        y = self.conv2(x)
        return y

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
