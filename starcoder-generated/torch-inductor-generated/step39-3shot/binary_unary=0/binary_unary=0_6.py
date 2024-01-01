
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.ops.aten._cudnn_convolutionFloat32(x1, x2, x2)
        v2 = torch.ops.aten.relu(v1)
        v3 = torch.ops.aten._cudnn_convolutionFloat32(v2, x2, x3)
        v4 = torch.ops.aten.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
