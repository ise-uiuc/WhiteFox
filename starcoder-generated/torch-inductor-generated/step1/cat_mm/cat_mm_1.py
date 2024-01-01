
class Model(torch.nn.Module):
    def forward(self, y, z):
        q = torch.cat([y, z], 3)
        return q

# Initializing the model
m = Model()

# Inputs to the model
y = torch.randn(1, 64, 56, 14)
z = torch.randn(1, 64, 56, 14)
