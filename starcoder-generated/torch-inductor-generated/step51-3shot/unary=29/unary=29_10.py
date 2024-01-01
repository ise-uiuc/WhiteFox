
class Model(torch.nn.Module):
    def __init__(self, min_value=0.8, max_value=-0.9):
        super().__init__()
        self.maxpool_with_argmax = torch.nn.MaxPool2d(3, stride=1, return_indices=False, ignore_indices=False, ceil_mode=False, return_indices=True)
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 2, 2, stride=2)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x2):
        v1 = self.conv_transpose(x2)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.maxpool_with_argmax(v3)[1]
        return v4
# Inputs to the model
x2 = torch.randn(1, 4, 20, 20)
