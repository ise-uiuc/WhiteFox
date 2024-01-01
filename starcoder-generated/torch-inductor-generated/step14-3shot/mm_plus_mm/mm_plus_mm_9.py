
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 128)
        self.linear2 = nn.Linear(128, 64)
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x1)
        return x1 + x2
# Inputs to the model
batchsize = 2
inputsize = 128
model = Model()
input = torch.randint(high = 64, size = (batchsize, inputsize)).float()
