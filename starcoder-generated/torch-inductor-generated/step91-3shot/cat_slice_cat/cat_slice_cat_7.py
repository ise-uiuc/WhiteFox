
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        l1 = [x1, x1]
        t1 = torch.cat(l1, dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:20]
        l2 = [t1, t3]
        t4 = torch.cat(l2, dim=1)
        return t4
 
# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
x2 = torch.randn(1, 3, 32, 32)
