
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1
        v3 = v2[:, 0:v2.size(1)]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 10, 10)
x2 = torch.randn(1, 1, 22, 15)
v = m(x1, x2)
size = torch.Tensor.size(v)
