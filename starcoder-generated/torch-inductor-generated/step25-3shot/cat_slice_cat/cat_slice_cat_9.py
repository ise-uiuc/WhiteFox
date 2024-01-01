
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17):
        v1 = torch.cat([x1, x2, x3, x4, x5])
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:x6.shape[1]]
        v4 = torch.cat([v1, v3])
        v5 = torch.cat([x7, x8, x9, x10, x11])
        v6 = v5[:, 0:9223372036854775807]
        v7 = v6[:, 0:x12.shape[1]]
        v8 = torch.cat([v5, v7])
        v9 = torch.cat([x13, x14, x15, x16, x17])
        v10 = v9[:, 0:9223372036854775807]
        v11 = v10[:, 0:x18.shape[1]]
        v12 = torch.cat([v9, v11])
        return v4, v8, v12

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn((1, 40, 5, 5))
x2 = torch.randn((1, 60, 5, 5))
x3 = torch.randn((1, 80, 5, 5))
x4 = torch.randn((1, 100, 5, 5))
x5 = torch.randn((1, 120, 5, 5))
x6 = torch.randn((1, 20, 5, 5))
x7 = torch.randn((1, 40, 5, 5))
x8 = torch.randn((1, 60, 5, 5))
x9 = torch.randn((1, 80, 5, 5))
x10 = torch.randn((1, 100, 5, 5))
x11 = torch.randn((1, 120, 5, 5))
x12 = torch.randn((1, 20, 5, 5))
x13 = torch.randn((1, 40, 5, 5))
x14 = torch.randn((1, 60, 5, 5))
x15 = torch.randn((1, 80, 5, 5))
x16 = torch.randn((1, 100, 5, 5))
x17 = torch.randn((1, 120, 5, 5))
x18 = torch.randn((1, 20, 5, 5))

