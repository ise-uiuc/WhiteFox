
class Model(torch.nn.Module):
    def __init__(self, loopVar, batch_size):
        super().__init__()
        self.loopVar = loopVar
        self.batch_size = batch_size
    def forward(self, x1, x2):
        v = []
        for loopVar6 in range(self.loopVar):
            for batchIdx2 in range(self.batch_size):
                v.append(torch.mm(x1[batchIdx2], x2[batchIdx2]))
        return torch.cat(v, 1)
loopVar = 1
batch_size = 1
# Inputs to the model
x1 = torch.randn(batch_size, 3, 3)
x2 = torch.randn(batch_size, 3, 3)
