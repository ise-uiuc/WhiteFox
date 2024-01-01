
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        f = x
        for i in range(5):
             f = torch.cat((f, f, f, f), dim=1)
             f = f.view(f.shape[0], -1)
             f = torch.sigmoid(f)
        return f + x
# Inputs to the model
x = torch.randn(2, 3, 4)
