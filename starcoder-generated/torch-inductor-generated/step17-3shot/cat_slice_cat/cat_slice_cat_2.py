
class Model(torch.nn.Module):
    def forward(self, x):
        size1 = torch.numel((x[:, 0, :, :]))
        t1 = torch.cat(x, dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:size1]
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
