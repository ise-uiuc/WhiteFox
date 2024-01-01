
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.ConvTranspose2d(3, 5, kernel_size=1, stride=1)
    def forward(self, in1):
        a1 = torch.sigmoid(self.layer1(in1))
        return a1
# Inputs to the model
in1 = torch.randn(1, 3, 128, 128) # A random input in the desired range
