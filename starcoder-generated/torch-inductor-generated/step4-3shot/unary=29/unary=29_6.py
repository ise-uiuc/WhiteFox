
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.19839155912338597, max_value=7.613178804127784):
        super().__init__()
        self.convtran_1 = torch.nn.ConvTranspose2d(3, 64, 4, stride=2, padding=1)
        self.convtran_1_2 = torch.nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)
        self.convtran_1_3 = torch.nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)
        self.relu = torch.nn.ReLU(True)
        self.tanh = torch.nn.Tanh()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.convtran_1(x1)
        v2 = self.convtran_1_2(v1)
        v3 = self.convtran_1_3(v2)
        v4 = torch.clamp(v3, self.min_value, self.max_value)
        v5 = self.relu(v4)
        v6 = self.tanh(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
