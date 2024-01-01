
class Model(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, input_tensors):
        x = torch.cat(input_tensors, dim=1)
        x = x[:, 0:9223372036854775807]
        x = x[:, 0:size]
        x = torch.cat([x, x], dim=1)
        return x

# Initializing the model
m = Model()

# Input tensor to the model
x1 = torch.randn(1, 320)
x2 = torch.randn(1, 384)
x3 = torch.randn(1, 320)
x = [x1, x2, x3]
