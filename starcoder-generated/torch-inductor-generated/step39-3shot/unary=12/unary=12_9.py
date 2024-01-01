
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = F.conv2d(x1, torch.randn(8, 3, 1, 1), bias=None, stride=1, padding=1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
