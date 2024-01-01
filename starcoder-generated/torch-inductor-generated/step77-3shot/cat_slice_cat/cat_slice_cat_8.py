
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
 
    def forward(self, *args):
        ls = [i for i in args]
        t1 = torch.cat(ls, dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:self.size]
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Initializing the model
size = int(np.random.randint(2, 10, 1))
