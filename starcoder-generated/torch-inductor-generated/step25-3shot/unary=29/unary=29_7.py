
class Model(torch.nn.Module):
    def __init__(self, max_value=torch.tensor(1.68)):
        super(Model, self).__init__()
        self.tanh = torch.nn.Tanh()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 1, 3, stride=2, padding=0)
        self.clamp_max = torch.nn.Hardtanh(max_value=max_value.item())
        self.output = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_max(v1, max_value)
        v3 = self.tanh(v2)
        v4 = self.output(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 24, 24)
