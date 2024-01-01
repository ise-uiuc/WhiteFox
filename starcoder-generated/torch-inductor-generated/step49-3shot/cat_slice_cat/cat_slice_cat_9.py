
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        v1 = [None]
        for i1 in range({1}):
            v1.insert(i1, x1)
        v2 = torch.cat(v1, dim=1)
        v3 = v2[:, 0:9223372036854775807]
        v4 = v3[:, 0:x2]
        v5 = torch.cat([v2, v4], dim=1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
x2 = torch.randint(-120, -110, [1], dtype=torch.int64)[0]
x3 = torch.randint(-110, -100, [1], dtype=torch.int64)[0]
