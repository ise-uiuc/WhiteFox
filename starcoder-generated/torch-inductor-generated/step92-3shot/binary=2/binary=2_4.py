
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1,4,1,stride=1,padding=0,bias=False)
    def forward(self, input):
        x = torch.cat((torch.randn(2, 1, 4, 4, device="cuda"), input), dim=1)
        return self.conv(x) - 0.2
# Inputs to the model
input = torch.randn(1, 1, 38, 38, device="cuda")
