
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 4, 3, stride=1, padding=1, output_padding = 0)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.relu(v)
        return v2

# Initializing the model
m = Model()

#Inputs to the model
x = torch.randn(1, 3, 8, 8)
