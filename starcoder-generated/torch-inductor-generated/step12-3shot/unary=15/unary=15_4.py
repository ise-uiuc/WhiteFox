
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = torch.nn.functional.conv2d(x1, torch.randn(8, 3, 1, 1), None, [2, 2], 1, [1, 1], False)
        v2 = torch.nn.functional.prelu(v1, torch.randn(8))
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
