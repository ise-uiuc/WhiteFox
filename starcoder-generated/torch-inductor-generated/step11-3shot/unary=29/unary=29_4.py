
class Model(torch.nn.Module):
    def __init__(self, min_value=-2.0, max_value=3):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(3, 7, kernel_size=(5, 5), stride=(5, 5), padding=(1, 1))
        self.max_value = max_value
        self.min_value = min_value
    def forward(self, x1):
        v1 = self.conv_transpose2d(x1)
        v2= torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
