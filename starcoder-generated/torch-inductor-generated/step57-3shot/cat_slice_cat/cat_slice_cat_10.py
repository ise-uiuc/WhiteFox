
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        x1_size = x1.size()
        x1_size_2 = x1_size[2] * x1_size[3]
        x2_size = x2.size()
        x2_size_2 = x2_size[2] * x2_size[3]
        v2 = torch.cat([x1, x2], dim=1)
        v3_1 = v2[:, 0:9223372036854775807]
        v3_2 = v3_1[:, 0:x1_size_2]
        v4 = torch.cat([v2, v3_2], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
