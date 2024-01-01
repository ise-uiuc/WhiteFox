
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, kernel=3):
        v1 = F.conv1d(x, kernel, stride=1, padding=1)
        v2 = v1.size(-1)
        v3 = torch.arange(v2)
        v4 = v3 + torch.floor_divide(v2, 2)
        v5 = {"other": v4}
        v6 = torch.nn.functional.relu(v5)
        v7 = v6 + v1
        return v7

# Initializing the model
m = Model()

# Inputs to the model
conv_kernel_1d = torch.randn((3, 5, 2))
x = torch.randn(5, 3, 64)
