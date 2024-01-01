
class Model(torch.nn.Module):
    def __init__(self, min_value=3.3967, max_value=8.3466):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(5, 27, 2, 2, output_padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 5, 89, 88)
