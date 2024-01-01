
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(39, 42, 3, stride=3, padding=1, output_padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x2):
        a1 = self.conv_t(x2)
        a2 = self.relu(a1)
        return torch.nn.functional.adaptive_avg_pool2d(a1, (2, 1))
# Inputs to the model
x2 = torch.randn(2, 39, 23, 9)
