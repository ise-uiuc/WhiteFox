
class Model(torch.nn.Module):
    def __init__(self, min_value=0.5, max_value=3):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.tanh2 = torch.nn.Tanh()
        self.conv2d_transpose = torch.nn.ConvTranspose2d(3, 3, 1, stride=1, padding=1)
    def forward(self, tensor):
        v1 = torch.clamp(tensor, self.min_value, self.max_value)
        v2 = self.tanh2(v1)
        v3 = self.conv2d_transpose(v2)
        return v3
# Inputs to the model.
tensor = torch.randn(1, 3, 64, 64)
