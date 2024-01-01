
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = [x1, x2]
        v2 = torch.cat(v1, dim=1)
        v3 = v2[:, 0:18446744073709551615]
        v4 = v3[:, 0:2]
        v5 = [v2, v4]
        v6 = torch.cat(v5, dim=1)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 12, 224, 224)
x2 = torch.randn(1, 22, 224, 224)
