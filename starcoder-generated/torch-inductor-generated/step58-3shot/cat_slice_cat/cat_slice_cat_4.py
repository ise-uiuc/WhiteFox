
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input_tensors1, size1):
        t1 = torch.cat(input_tensors1, dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:size1]
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Initializing the model
m = Model()

# Inputs to the model
input_tensors1 = tuple(torch.randn(1, i, 64, 64) for i in range(5))
size1 = torch.randint(low=1, high=5, size=(), dtype=torch.int64)
