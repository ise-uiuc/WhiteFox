
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=0)
    def forward(self, x):
        t1 = self.conv(x) # t1 is a B x 3 x 32 x 32 tensor
        t2 = t1 - 0.5 # t2 is a B x 3 x 32 x 32 tensor
        return t2 
# Inputs to the model
x = torch.randn(1, 3, 100, 100)
