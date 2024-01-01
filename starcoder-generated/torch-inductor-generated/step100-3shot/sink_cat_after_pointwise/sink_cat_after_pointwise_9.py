
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(6, 5)
        self.linear2 = torch.nn.Linear(7, 8)
    def forward(self, x):
        if (x.shape[1] == 8):
            x = x.transpose(1, 2)
        z = self.linear2(x)
        z = z.reshape(8, 4, 3, 2)
        y = self.linear1(z)
        return y
# Inputs to the model
x = torch.randn(1, 1, 8, 6)
