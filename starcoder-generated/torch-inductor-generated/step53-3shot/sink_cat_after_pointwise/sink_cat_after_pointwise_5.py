
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, x), dim=0)
        return x.permute(0, 2, 1) if x.shape!= (2, 2) else x.permute(0, 2, 1).tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)
