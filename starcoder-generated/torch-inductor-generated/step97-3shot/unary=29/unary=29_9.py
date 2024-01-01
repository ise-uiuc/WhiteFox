
class Model(torch.nn.Module):
    def __init__(self, min_value=-128.754, max_value=-125.311):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(1, 1280, 3, stride=1, padding=1)
        self.conv_transpose_prelu = torch.nn.PReLU(1280)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(1280, 1, 2, stride=1, padding=0)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v1_prelu = self.conv_transpose_prelu(v1)
        v2 = self.conv_transpose2(v1_prelu)
        v3 = torch.sigmoid(v2)
        v3_relu = self.relu(v3)
        v4 = torch.clamp_min(v3_relu, self.min_value)
        v5 = torch.clamp_max(v4, self.max_value)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
