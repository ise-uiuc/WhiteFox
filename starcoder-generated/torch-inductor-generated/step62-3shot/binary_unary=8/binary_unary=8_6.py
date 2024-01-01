
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1000, 1)
        self.layernorm1 = torch.nn.LayerNorm((1000,), eps=1e-05, elementwise_affine=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.layernorm1(v1)
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 128, 128)
