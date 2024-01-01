
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2, padding=0, output_padding=0, groups=1, bias=False, dilation=1)
 
    def forward(self, x):
        v32 = self.conv_t(x)
        v33 = torch.nn.functional.relu(v32)
        return v33

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16, 16, 16)
