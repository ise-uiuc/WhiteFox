
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 32, kernel_size=(1, 24), stride=(1, 2), padding=(0, 12))
    def forward(self, input):
        return self.conv_transpose(input).clamp(self.min_value, self.max_value)
# Inputs to the model
input = torch.randn(1, 3, 100, 250)
