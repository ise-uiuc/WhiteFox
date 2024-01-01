
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, *input_tensors):
        t1 = torch.cat(tuple(input_tensors), dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:51]
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Inputs to the model
inputs = []
for _ in range(4):
    inputs.append(torch.randn(1, 51, 20, 20))
