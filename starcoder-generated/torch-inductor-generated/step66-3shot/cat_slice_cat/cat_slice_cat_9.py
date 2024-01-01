
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5):
        _tmp = torch.cat([x1, x2], dim=1)
        v1 = _tmp[:, 0:9223372036854775807]
        v2 = v1[:, 0:471859]
        _tmp = torch.cat([_tmp, v2], dim=1)
        v3 = _tmp[:, 0:471859]
        _tmp = torch.cat([v3, x3], dim=1)
        v4 = _tmp[:, 0:1431655765]
        _tmp = torch.cat([_tmp, x4], dim=1)
        v5 = _tmp[:, 0:2863311530]
        _tmp = torch.cat([_tmp, x5], dim=1)
        v6 = _tmp[:, 0:471859]
        _tmp = torch.cat([v4, v5, v6], dim=1)
        t1 = _tmp[:, 0:9223372036854775807]
        t2 = t1[:, 0:9223372036854775807]
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 30, 14)
x2 = torch.randn(1, 31, 16)
x3 = torch.randn(1, 35, 15)
x4 = torch.randn(1, 32, 17)
x5 = torch.randn(1, 40, 16)
