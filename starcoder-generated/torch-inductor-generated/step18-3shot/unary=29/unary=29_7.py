
class Model(torch.nn.Module):
    def __init__(self, min_value=0.22, max_value=5.2):
        super().__init__()
        self.leaky_relu = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(p=0.38)
        self.conv_transpose = torch.nn.ConvTranspose1d(4, 2, 1, stride=2)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        v1 = self.conv_transpose(x)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.leaky_relu(v3)
        v5 = self.dropout(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 4, 32)
