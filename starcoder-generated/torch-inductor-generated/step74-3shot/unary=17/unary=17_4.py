
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()
        self.conv2d = torch.nn.Conv2d(2, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    def forward(self, input):
        v0 = input
        v1 = torch.transpose(v0, 2, 3)
        v2 = self.conv2d(v1)
        v3 = v2.permute(0, 2, 1, 3)
        v4 = self.tanh(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 2, 7, 7)
