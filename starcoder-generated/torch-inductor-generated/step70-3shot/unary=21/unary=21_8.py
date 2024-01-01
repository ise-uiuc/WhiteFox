
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.v1 = torch.nn.Conv2d(16, 13, 7, 1)
        # self.v2 = torch.nn.Conv2d(13, 26, 1, 1)
        # self.v3 = torch.nn.Conv2d(int(max(13//23,1)), 1, 1, 1)
        self.v4 = torch.nn.Conv2d(53, 31, 1, 1)
    def forward(self, x):
        # x = self.v1(x)
        # x = self.v2(x)
        # x = torch.tanh(x)
        # x = self.v3(x)
        x = self.v4(x)
        return x

# Inputs to the model
x = torch.randn(64, 53, 32, 32)
