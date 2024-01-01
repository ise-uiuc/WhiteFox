
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.cat = torch.nn.modules.rnn.RNN(9, 9, 2, batch_first=True)
        self.min = min
        self.max = max
    def forward(self, x1, x2):
        v1 = torch.cat((x1, x2), 1)
        v2 = self.cat(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 1e+10
max = 0.1
# Inputs to the model
x1 = torch.randn(6, 9, 5)
x2 = torch.randn(6, 9, 5)
