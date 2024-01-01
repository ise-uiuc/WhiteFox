
class Model(torch.nn.Module):
    def forward(self, inputs):
        x, y, z = inputs
        x1 = x * y
        x2 = x + z
        y1 = x * z
        z1 = y * z
        o1 = (x1 + x2) * z
        o2 = (y1 + z1) * x
        o3 = o1 + o2
        return o3
# Inputs to the model
inputs = [torch.randn(1), torch.randn(1), torch.randn(1)]

