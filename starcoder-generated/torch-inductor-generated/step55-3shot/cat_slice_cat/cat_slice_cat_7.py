
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        v0 = self.conv1(x3)
        v1 = self.conv2(x2)
        v2 = self.conv3(x1)
        v3 = torch.cat([v0, v2], dim=1)
        v4 = v3[:, :9223372036854775807]
        v5 = v4[:, :x2.size(2)]
        v6 = torch.cat([v3, v5], dim=1)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 56, 56)
x2 = torch.randn(1, 8, 24, 24)
x3 = torch.randn(1, 4, 24, 24)
