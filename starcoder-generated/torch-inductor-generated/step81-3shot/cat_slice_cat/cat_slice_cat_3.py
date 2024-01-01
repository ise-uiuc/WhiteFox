
class Model(torch.nn.Module):
    def forward(self, x, y):
        t1 = torch.cat([x, y], dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:x.size(3)]
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(size=(1, 10, 5, 4))
y = torch.randn(size=(1, 20, 5, 4))
