
class Model(torch.nn.Module):
    def __init__(self, min_value=0.5, max_value=10.9):
        super().__init__()
        self.linear = torch.nn.Linear(2, 28)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 4, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, input):
        v1 = self.conv_transpose(input)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.sigmoid(v3)
        v5 = self.linear(v4)
        v6 = v5.view(v5.size() + (1, 1))
        return v6
# Inputs to the model
input = torch.randn(1, 3, 64, 64)
