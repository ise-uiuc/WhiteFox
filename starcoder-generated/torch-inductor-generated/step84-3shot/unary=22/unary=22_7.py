
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
                        # 3 * 64 * 64 -> 2048
                        torch.nn.Conv2d(3, 2048, 1, 1, 0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
                        torch.nn.ReLU(),
                        )
 
    def forward(self, x1):
        v1 = self.layers(x1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
