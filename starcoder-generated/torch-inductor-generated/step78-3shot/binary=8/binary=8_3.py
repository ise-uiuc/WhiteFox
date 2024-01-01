
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        return v1
class MyFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, x): # x is the input to the model
        # Return PyTorch function to symbolically trace into this forward function.
        return None
    @staticmethod
    def forward(ctx, x):
        # Return the output of the PyTorch module. We use the ctx parameter to store tensors used in the computation here.
        return None
