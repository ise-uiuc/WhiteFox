
class Model(torch.nn.Module):
    def forward(self, **kwargs):
        t1 = torch.cat(list(kwargs.values()), dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:__SIZE__]
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Model inputs
x1 = torch.randn(2, 10, 32, 32)
x2 = torch.randn(2, 10, 32, 32)
