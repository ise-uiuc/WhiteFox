
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = m1()
    def forward(self, x1):
        x2 = self.m1(x1)
        return torch.relu(x2)
# Inputs to the model
import torchvision
x1 = torch.tensor([[[[  0.,   1.,   2.],
                     [-29.,  -2.,  -1.]],

                    [[  3.,   4.,   5.],
                     [  6.,   7.,   8.]],
                    
                    [[  9.,  10.,  11.],
                     [ 12.,  13.,  14.]]]])
