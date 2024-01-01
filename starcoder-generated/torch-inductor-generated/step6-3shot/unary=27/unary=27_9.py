
class Model(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()

    def forward(self, x1):
        weight = torch.randint(low=-2, high=2, size=(3, x1.shape[1], 3, 3)).float()
        v1 = torch.nn.functional.conv2d(x1, weight, stride=(1, 1), padding=2, dilation=2)
        v2 = torch.unsqueeze(v1, dim=-1)
        v3 = torch.clamp_min(v2, 0.01)
        return v3


# Inputs to the model
x1 = torch.randn(16, 5, 115, 84)
alpha = 0.1
