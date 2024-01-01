
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2).cuda()
    def forward(self, x3):
        v6 = x3.cuda()
        v4 = self.linear(v6)
        v5 = v4.permute(0, 2, 1).cuda()
        return v5
# Inputs to the model
x3 = torch.randn(1, 2, 2)
