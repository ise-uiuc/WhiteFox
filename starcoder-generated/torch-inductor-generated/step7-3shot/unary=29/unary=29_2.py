
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.1757653801064682, max_value=-0.0011746449844745787):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(3, 8, kernel_size=(3, 2), stride=(1, 1), padding=(1, 0))
        self.relu = torch.nn.ReLU()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 8, 1, stride=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose2d(x1)
        v2 = self.relu(v1)
        v3 = self.conv_transpose(v2)
        v4 = torch.clamp_max(v3, self.max_value)
        v5 = torch.clamp_min(v4, self.min_value)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 5, 4)
