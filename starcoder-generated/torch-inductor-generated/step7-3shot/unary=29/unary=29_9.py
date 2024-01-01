
class Model(torch.nn.Module):
    def __init__(self, min_value=0.1, max_value=0.3):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1)
        self.max_value = max_value
        self.min_value = min_value
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.tanh(v1)
        v3 = v2.clamp(self.min_value, self.max_value)
        v4 = self.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
