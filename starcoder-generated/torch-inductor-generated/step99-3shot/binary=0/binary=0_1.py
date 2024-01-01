
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 1, stride=1, padding=1)
    def forward(self, x1, pad_mode="valid"):
        if pad_mode == "valid":
            padding1 = [1, 1]
        elif pad_mode == "same":
            padding1 = [0, 0]
        v1 = self.conv(x1, padding1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 32, 32)
