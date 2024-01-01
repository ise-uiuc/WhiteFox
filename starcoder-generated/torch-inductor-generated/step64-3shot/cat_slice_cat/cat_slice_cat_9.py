
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        c1 = torch.cat([x1, x2, x3], dim=1)
        s1 = c1[:, 0:9223372036854775807]
        s2 = s1[:, 0:5]
        s3 = torch.cat([c1, s2], dim=1)
        return s3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 8, 1)
x2 = torch.randn(3, 4, 1)
x3 = torch.randn(3, 2, 1)
