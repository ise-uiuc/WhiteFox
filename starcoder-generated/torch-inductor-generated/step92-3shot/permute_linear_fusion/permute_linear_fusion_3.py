
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = x1.permute(1, 0, 2).unsqueeze(1)
        x2 = torch.randn(2, 2, 3, 3)
        v1 = torch.nn.functional.conv2d(x1, x2, None)
        x3 = v1.squeeze(1)
        y1 = torch.pow(x3, 2)
        return torch.nn.functional.linear(y1, x3, None)
# Inputs to the model
x1 = torch.randn(2, 2)
