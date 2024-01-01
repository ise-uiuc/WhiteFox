
class Model(torch.nn.Module):
    def __init__(self, min_value=-2, max_value=-8):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(1, 3, (3, 5))
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 1, 3)
        self.sigmoid = torch.nn.Sigmoid()
        self.avgpool = torch.nn.AvgPool2d(1, stride=1, padding=2)
        self.batch_norm = torch.nn.BatchNorm1d(num_features=1, eps=1.0e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(3, 1, 1)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(3, stride=1, padding=2)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.maxpool(v1)
        v3 = self.conv_transpose(v2)
        v4 = self.batch_norm(v3)
        v5 = self.sigmoid(v4)
        v6 = self.avgpool(v5)
        v7 = self.conv_transpose2(v6)
        v8 = torch.clamp_max(v7, self.max_value)
        v9 = self.relu(v8)
        v10 = torch.clamp_min(v9, self.min_value)
        return v10
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
