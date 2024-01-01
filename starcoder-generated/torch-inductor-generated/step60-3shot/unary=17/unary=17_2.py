
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 3, (3, 2), stride=(2, 1), output_padding=(1, 0), bias=False)
        self.conv2 = torch.nn.ConvTranspose2d(3, 3, (3, 1), stride=(1, 2), padding=(1, 0), output_padding=(0, 0), bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 40, 20)
