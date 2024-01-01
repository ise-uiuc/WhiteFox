
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:517121]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
size = torch.randint(1, 128, (1,), dtype=torch.int64)
x1 = torch.randn(1, int(2.6e+08), 64, 64)
x2 = torch.randn(1, int(1.3e+07), 64, 64)
