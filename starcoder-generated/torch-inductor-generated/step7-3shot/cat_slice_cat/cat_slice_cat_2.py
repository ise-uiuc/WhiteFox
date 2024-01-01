
class Model(torch.nn.Module):
    def forward(self, x1):
        t1 = torch.cat(x1, dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:x1.size(2)]
        t4 = torch.cat(x1 + t1, dim=1)
        return t4
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
