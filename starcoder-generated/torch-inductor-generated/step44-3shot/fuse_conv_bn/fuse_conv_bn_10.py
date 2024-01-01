
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.layer = torch.nn.Sequential(torch.nn.Conv3d(3, 4, 2, bias=True), torch.nn.Conv3d(3, 5, 3, bias=False))
    def forward(self, x2):
        s = self.layer(x2)
        return s
# Inputs to the model
x2 = torch.randn(1, 3, 4, 4, 4)
