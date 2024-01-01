
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(1, 1, 1)
        self.group0 = torch.nn.Sequential(
            (torch.nn.Conv2d(2, 2, 1),torch.nn.Conv2d(2, 2, 1)), # conv module with no weight
            (torch.nn.Conv2d(2, 2, 1,),torch.nn.Conv2d(2, 2, 1),torch.nn.Conv2d(2, 2, 1)) # conv module with multiple conv module as its sub modules
        )
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = self.group0(x1)
        return v1 + v2
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
