
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x, y):
        v1 = torch.cat(input_tensors, dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:size]
        return torch.cat([v1, v3], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(4, 512, 7, 7)
y = torch.randn(4, 2048, 7, 7)
