
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t = x
        y = torch.cat((t.unsqueeze(0), x.unsqueeze(0), x), dim=0)
        return y.view(y.shape[0], -1).relu()
# Inputs to the model
x = torch.ones(1, 20)
