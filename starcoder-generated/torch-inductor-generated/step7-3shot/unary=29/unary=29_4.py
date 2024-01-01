
class Model(torch.nn.Module):
    def __init__(self, min_value=0.024511211144091892, max_value=0.09386511436004639):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
        self.conv2d = torch.nn.Conv2d(8, 3, 1, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp(v1, self.min_value, self.max_value)
        v3 = self.conv2d(v2)
        v4 = self.relu(v3)
        v5 = self.softmax(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 112, 112)
