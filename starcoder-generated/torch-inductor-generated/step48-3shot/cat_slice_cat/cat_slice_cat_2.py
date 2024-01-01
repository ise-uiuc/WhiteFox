
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
 
    def forward(self, *args):
        n = len(args) // 2
        l1, l2 = args[:n], args[n:]
        t1 = torch.cat(l1, dim=1)
        t3 = t1[:, :, self.size:]
        t4 = torch.cat((t1, t3), dim=1)
        return t4

# Initializing the model
size = 13
model = Model(size)

# Inputs to the model
x1 = torch.randn(1, 1, 64, 52)
x2 = torch.randn(1, 1, 64, 34)
