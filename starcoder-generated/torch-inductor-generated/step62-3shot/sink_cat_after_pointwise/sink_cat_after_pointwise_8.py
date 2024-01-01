
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a = torch.cat((torch.cat((x.permute(0, 2, 1), x),dim=1), x.detach()), dim=-1)
        return a if a.shape[1] >= 0 else a + 0
# Inputs to the model
x = torch.randn(2, 2, 2)
