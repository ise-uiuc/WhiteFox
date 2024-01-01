
class Model(torch.nn.Module):
    def __init__(self, min_value=5.8, max_value=2.1):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.conv_transpose = torch.nn.ConvTranspose3d(1, 1, 1, stride=1)
        self.tanh_ = torch.nn.Tanh()
        self.relu_ = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        # First, do the first operation, then the second, and then the first again.
        # Also do the second, then the first, and then the second again.
        v1 = self.conv_transpose(x)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.tanh_(v3)
        v5 = self.relu_(v4)
        v6 = self.softmax(v5)
        return v6
# Inputs to the model
x = torch.randn(3, 1, 4, 4)
