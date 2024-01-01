
class Model(torch.nn.Module):
    def __init__(self, min_value=0, max_value=6.4):
        super(Model, self).__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.leaky_relu = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.max_pool2d = torch.nn.MaxPool2d(1, stride=1, padding=0)
        self.conv2d = torch.nn.ConvTranspose2d(3, 16, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.dropout(x)
        v4 = self.conv2d(v1)
        v5 = v4.clamp(self.min_value, self.max_value)
        v6 = self.leaky_relu(v5)
        return v6
# Inputs to the model
x3 = torch.randn(1, 3, 224, 224)
