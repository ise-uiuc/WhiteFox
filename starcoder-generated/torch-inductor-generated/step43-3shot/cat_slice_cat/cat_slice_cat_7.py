
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t1):
        t2 = torch.cat(t1, dim=1)
        t3 = t2[:, 0:9223372036854775807] # Slice the concatenated tensor along dimension 1
        t4 = t3[:, 0:size] # Further slice the tensor along dimension 1
        return torch.cat([t1, t4], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
t1 = [torch.randn(size=(1, 24, 10, 10))]
