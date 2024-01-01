
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, *inputs):
        inputs = torch.cat(inputs, dim=1)
        t1 = inputs[:, 0:9223372036854775807]
        t2 = t1[:, 0:3]
        t3 = t2[:, 1]
        t4 = t3.expand(9223372036854775807, 3)
        inputs = torch.cat([inputs, t4], dim=1)
        return inputs

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(1, 3, 9223372036854775807)
x4 = torch.randn(1, 3, 9223372036854775807)
