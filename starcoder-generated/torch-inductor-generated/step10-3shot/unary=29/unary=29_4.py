
class Model(torch.nn.Module):
    def __init__(self, min_value=0, max_value=6.4):    
        super(Model, self).__init__()
        self.leaky_relu = torch.nn.LeakyReLU()
        self.max_pool2d = torch.nn.MaxPool2d(1, stride=1, padding=0)
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 32, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x3):
        v1 = self.conv_transpose(x3)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.leaky_relu(v3)
        return v4
min_value = 0
max_value = 3
# Inputs to the model
x3 = torch.randn(1, 3, 64, 64)
