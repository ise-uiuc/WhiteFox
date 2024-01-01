
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Conv2d(3, 3, 3, bias=False)
        self.layer2 = torch.nn.BatchNorm2d(3)
    def forward(self, x):
# The code snippet below shows that self.layer1(x) takes the input x as its input, and outputs a tensor of shape (1, 3, 3, 3). 
# Since that output is consumed by self.layer2(s1), the pattern is applied
        s1 = self.layer1(x)
        s1 = self.layer2(s1)
        return s1
# Inputs to the model
x = torch.randn(1, 3, 5, 5)
