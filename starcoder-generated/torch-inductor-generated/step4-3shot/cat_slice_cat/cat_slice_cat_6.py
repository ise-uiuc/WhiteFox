
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8):
        v1 = torch.cat((x1, x2))
        v2 = v1[:, ::2]
        v3 = torch.cat((v1, v2))
        v4 = v3[:, :-87648823]
        v5 = torch.cat((v4, v2, v3))
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 32, 64, 64)
x4 = torch.randn(1, 64, 64, 64)
x5 = torch.randn(1, 128, 64, 64)
x6 = torch.randn(1, 256, 64, 64)
x7 = torch.randn(1, 256, 64, 64)
x8 = torch.randn(1, 512, 64, 64)
