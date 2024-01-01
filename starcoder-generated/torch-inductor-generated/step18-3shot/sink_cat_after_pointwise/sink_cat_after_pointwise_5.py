
class Model(torch.nn.Module):
    def __init__(self, tch):
        super().__init__()
        self.tch = tch
    def forward(self, x):
        x = torch.cat((x, self.tch), dim=1) if x.shape!= self.tch.shape else x
        y = torch.tanh(x.view(x.shape[0], -1))
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
tch = torch.randn(1, 6)
