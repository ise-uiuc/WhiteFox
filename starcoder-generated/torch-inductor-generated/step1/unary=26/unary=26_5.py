
M = torch.nn.ConvTranspose2d(99, 128, kernel_size=[4, 4], 
stride=[4, 4])
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = M
        self.relu = torch.nn.ReLU()

    def forward(self, x, negative_slope=0.3):
        v1 = self.conv(x)
        v2 = v1 > 0
        relu = self.relu(v1)
        v3 = torch.where(v2, v1, relu * negative_slope)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
M_init=m.conv.weight.detach().clone()
x = torch.randn(1, 99, 197, 189)

