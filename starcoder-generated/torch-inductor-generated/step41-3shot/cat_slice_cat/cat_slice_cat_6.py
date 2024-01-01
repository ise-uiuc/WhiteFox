
class Model(torch.nn.Module):

    def forward(self, x1, x2, x3, x4):
        v1 = torch.cat(input_tensors, dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:size]
        return torch.cat([v1, v3], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
x2 = torch.randn(1, 1, 28, 28)
x3 = torch.randn(1, 1, 28, 28)
x4 = torch.randn(1, 1, 28, 28)

