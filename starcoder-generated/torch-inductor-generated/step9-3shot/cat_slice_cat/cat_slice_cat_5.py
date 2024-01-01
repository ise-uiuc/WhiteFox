
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x5):
        v5 = torch.cat(x5, dim=1)
        v6 = v5[:, 0:]
        v7 = v6[:, :9223372036854775807]
        v8 = v7[:, 0:v6.size(2)]
        v9 = torch.cat([v5, v8], dim=1)
        return v9

# Initializing the model
m = Model()


# Inputs to the model

# Dimensions of all the input tensors must be same
x5 = []
x5.append(torch.randn(1, 1, 235929600000000000, 150))
x5.append(torch.randn(1, 1, 70945146666666666, 150))
