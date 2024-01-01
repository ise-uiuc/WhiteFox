
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 6, kernel_size=(2, 3), stride=(1, 3), padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.relu(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 23, 75)
