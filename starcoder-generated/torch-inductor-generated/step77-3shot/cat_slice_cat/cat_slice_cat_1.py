
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        d = torch.cat([x1, x2], dim=1)
        s1 = d[:,0:9223372036854775807]
        s2 = s1[:,0:x1.shape[1] + x2.shape[1]]
        d1 = torch.cat([d, s2], dim=1)
        return d1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 5, 2, 2)
x2 = torch.randn(2, 7, 2, 2)
