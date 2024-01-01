
class Model(torch.nn.Module):
    def __init__(self, min_value=0, max_value=2):
        super().__init__()
        self.convt = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1, bias=False)
        self.relu = torch.nn.ReLU()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        x = self.convt(x)
        x = self.relu(x)
        x = torch.clamp(x, self.min_value, self.max_value)
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
