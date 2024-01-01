
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, l1, l2):
        t1 = torch.cat(l1, dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:len(l2)]
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Initializing the model
m = Model()

# Inputs to the model
l1 = [torch.randn(1, 3, 256, 512)]
__input_tensors__ = l1.extend(torch.randn(1, 3, 64, 64))
