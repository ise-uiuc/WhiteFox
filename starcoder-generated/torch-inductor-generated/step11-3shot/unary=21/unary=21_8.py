
class ModelTanh(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.module1 = torch.nn.Sequential(*(torch.nn.Conv2d(in_channels=i, out_channels=i,
                        kernel_size=3, stride=1, padding=1)
                        for i in range(0, 8)))
    def forward(self, x):
        v1 = self.module1(x)
        v2 = torch.tanh(v1)
        return v2

# Inputs to the model
x = torch.randn(1, 64, 64, 64)
