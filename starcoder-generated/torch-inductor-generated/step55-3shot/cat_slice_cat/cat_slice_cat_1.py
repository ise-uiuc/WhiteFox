
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x3, x4):
        v1 = x3.contiguous()
        v2 = x4.contiguous()
        v3 = torch.cat([v1, v2], dim=1)
        v4 = v3[:, 0:9223372036854775807]
        v5 = v4[:, 0:v2.size(1)]
        v6 = torch.cat([v3, v5], dim=1)
        return v6


# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(2, 10, 20, 30)
x4 = torch.randn(2, 10, 20, 40)
