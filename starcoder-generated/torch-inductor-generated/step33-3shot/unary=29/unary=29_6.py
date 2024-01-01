
class Model(torch.nn.Module):
    def __init__(self, min_value=0, max_value=32):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(4, 4, 1, stride=1, padding=0)
        self.conv2d = torch.nn.Conv2d(4, 2, 2, stride=2, padding=0)
        self.relu = torch.nn.ReLU()
        self.add = torch.nn.Add()
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.relu(v1)
        v3 = self.max_pool2d(v2)
        v4 = self.conv2d(v3)
        v5 = self.relu(v4)
        v6 = self.add(v2, v5)
        v7 = self.relu(v6)
        v8 = self.conv2d(v7)
        v9 = self.relu(v8)
        v10 = self.conv_transpose2d(v9)
        v11 = torch.clamp_min(v10, self.min_value)
        v12 = torch.clamp_max(v11, self.max_value)
        v13 = self.conv2d(v12)
        v14 = self.relu(v13)
        v15 = self.conv2d(v14)
        v16 = self.relu(v15)
        v17 = self.max_pool2d(v16)
        return v17
# Inputs to the model
x1 = torch.randn(1, 4, 65, 40)
