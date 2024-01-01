
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
 
    def forward(self, x_list1):
        v1 = torch.cat(x_list1, dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model(size)

# Inputs to the model
x1 = torch.randn(9, 3, 16, 16)
x_list1 = []
for _ in range(9):
    x_list1.append(x1)
