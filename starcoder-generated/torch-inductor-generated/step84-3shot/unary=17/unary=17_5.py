
class ReshapeModule(torch.nn.Module):
    def forward(self, t1, t2):
        sizes = [int(t2.shape[2] / t1.shape[2]), int(t2.shape[3] / t1.shape[3])]
        t3 = t2.reshape(t1.shape[0], t1.shape[1] * sizes[0] * sizes[1], t2.shape[2], t2.shape[3])
        t4 = torch.transpose(t3, 1, 2)
        return t4

# Inputs to the model
x1 = torch.randn(1, 12, 64, 64)
x2 = torch.randn(1, 8, 128, 128)

m = ReshapeModule()
o = m.forward(x1, x2)

print(o.shape)
