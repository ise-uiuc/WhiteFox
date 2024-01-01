
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        _tmp = torch.cat([x1, x2], dim=1)
        v1 = _tmp[:, 0:9223372036854775807]
        v2 = v1[:, 0:9223372036854775807]
        v3 = _tmp[:, 0:9223372036854775807]
        v6 = torch.cat([_tmp, v3], dim=1)
        return v6

# Initializing the model
m = Model()

# Input tensors to the model
x1 = torch.randn(1, 9223372036854775807, 512, 512)
x2 = torch.randn(1, 9223372036854775807, 512, 512)
