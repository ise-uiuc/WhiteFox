
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = x[:, 0:9223372036854775807]
        x = x[:, 0:128]
        x = torch.cat([x1, x], dim=1)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 114218704, 128)
x2 = torch.randn(1, 114218832, 128)
