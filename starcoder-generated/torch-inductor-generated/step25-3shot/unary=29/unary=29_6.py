
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax2d()
    def forward(self, x6):
        v1 = self.softmax(x6)
        v6 = torch.clamp_max(v1, max_value)
        return v6
        