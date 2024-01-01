 parameters
n_dim = 10 # The number of elements in weight
n_filter = 10 # The number of filters
kernel_size = (4, 4) # The size of the convolution kernel
stride = (1, 1) # The stride of the convolution

# Model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 8, kernel_size, stride)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 1
        v3 = torch.nn.functional.avg_pool3d(v2, kernel_size, stride)
        v4 = v3 / 8
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 32, 32, 32)
