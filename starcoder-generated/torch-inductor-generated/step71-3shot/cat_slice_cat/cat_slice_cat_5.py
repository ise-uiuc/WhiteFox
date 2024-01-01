
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.cat([x1, x2, x3], dim=1)
        __split_1__ = 2
        v2 = v1[:, 0:__split_1__]
        __split_2_size__ = 9223372036854775807 - __split_1__
        v3 = v2[:, 0:__split_2_size__]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, [3788127617580915200, 2514267141763963000])
x2 = torch.randn(1, [3999174358034427840, 6093890206561413180])
x3 = torch.randn(1, [1952590585167797900, 5015110992428644750])
x4 = torch.randn(1, [4225881722215062880, 560870384442868710])
x5 = torch.randn(1, [4149272535427806600])
