
class Model(torch.nn.Module):
    def __init__(self, min_value=12, max_value=torch.tensor([33.0], dtype=torch.float32)):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        y1 = v1.clamp(self.min_value, self.max_value.item())
        return y1
# Inputs to the model
x1 = torch.randn(4, 6, 1)
