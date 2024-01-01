
class Model(torch.nn.Module):
    def __init__(self):
        pass
 
    def forward(self, input_tensors):
        t1 = torch.cat(input_tensors, dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:size]
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Initializing the model
m = Model()

# Inputs to the model
input_tensors = []
for _ in range(10):
    input_tensors.append(torch.randn(3, 1, 255, 255))
