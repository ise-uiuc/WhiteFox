
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = [x1, x2]
        v2 = torch.cat(v1, dim=1)
        v3 = v2[:, 0:9223372036854775807]
        v4 = v3[:, 0:size]
        v5 = [v2, v4]
        v6 = torch.cat(v5, dim=1)
        return v6

# Initializing the model
m = Model(14)

# Inputs to the model
x2 = torch.randn(1, 10, 9, 9)
x3 = torch.randn(1, 14, 7, 7)
