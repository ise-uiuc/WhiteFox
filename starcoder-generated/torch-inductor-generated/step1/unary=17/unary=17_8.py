
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_trans = torch.nn.ConvTranspose2d(8, 3, 3, stride=2)
 
    def forward(self, x):
        return torch.nn.functional.relu(self.conv_trans(x))

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8, 128, 128)
