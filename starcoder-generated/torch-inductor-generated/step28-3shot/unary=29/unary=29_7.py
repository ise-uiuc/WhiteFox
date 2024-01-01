
class Model(torch.nn.Module):
    def __init__(self, min_value=0, max_value=1):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 2, 3, stride=1)
        self.add = torch.nn.Add()
        self.max_pool2d = torch.nn.MaxPool2d(7, stride=1, padding=3)
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 3, 1, stride=1, padding=1)
        self.softsign = torch.nn.Softsign()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        v1 = self.conv2d(x)
        v2 = self.conv2d(x)
        v3 = self.add(v1, v2)
        v4 = self.max_pool2d(v3)
        v5 = self.conv_transpose(v4)
        v6 = torch.clamp_min(v5, self.min_value)
        v7 = torch.clamp_max(v6, self.max_value)
        v8 = self.softsign(v7)
        return torch.flatten(v8, 1)
# Inputs to the model
x = torch.randn(1, 3, 112, 112)
