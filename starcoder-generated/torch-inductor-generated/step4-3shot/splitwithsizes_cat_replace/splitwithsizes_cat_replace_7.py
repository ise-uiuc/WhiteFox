
class Model(torch.nn.Module):
    def forward(self, x1):
        v2_0, v2_1, v2_2 = torch.split(x1, [3, 5, 2], dim=1)
        v3_0 = v2_0 * 1
        v4_0 = v2_1 + 23
        v5_0 = v4_0 * 1
        v6_0 = v5_0 + v3_0
        v7_0 = v2_2 + v6_0
        v8_0 = torch.cat([v7_0, v4_0, v5_0], dim=1)
        return v8_0

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 14, 64, 64)
