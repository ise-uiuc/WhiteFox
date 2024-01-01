
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 16, 5, stride=1, padding=1, bias=True, dilation=2)
        self.dropout = torch.nn.Dropout2d()
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v1 = self.dropout(v1)
        v2 = self.relu(v1)
        v2 = self.conv_transpose(v2)
        v1 = self.relu(v2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 16, 28, 28)
