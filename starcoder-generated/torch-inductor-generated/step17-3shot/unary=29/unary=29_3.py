
class Model(torch.nn.Module):
    def __init__(self, min_value=0, max_value=6.4):
        super(Model, self).__init__()
        self.leaky_rel = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(0.10000000000000001)
        self.max_pool2d = torch.nn.MaxPool2d(1, stride=1, padding=0)
        self.relu = torch.nn.ReLU6()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x3):
        v1 = self.dropout(x3)
        v6 = self.conv_transpose(x3)
        v2 = torch.clamp_min(v6, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.leaky_rel(v3)
        v5 = self.relu(v4)
        return v5
# Inputs to the model
x3 = torch.randn(1, 3, 224, 224)
