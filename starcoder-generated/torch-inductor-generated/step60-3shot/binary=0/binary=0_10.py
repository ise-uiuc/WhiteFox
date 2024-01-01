
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = torch.nn.Conv2d(5, 6, 1, stride=1, padding=1)
        self.model2 = torch.nn.ConvTranspose2d(6, 9, 1, stride=1, padding=1)
    def forward(self, x1, other=None):
        v1 = self.model1(x1)
        if other == None:
            other = torch.randn(v1.shape).to(v1.device)
        v2 = self.model2(other)
        v3 = v1 + v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64).to('cpu')
