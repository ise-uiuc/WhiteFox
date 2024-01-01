
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = None
 
    def forward(self, input, target, bias):
        return input + bias[None,:,None,None]
        
# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(4, 512, 8, 8)
target = torch.randn(4, 512, 8, 8)
bias = torch.nn.Parameter(torch.randn(1, 512, 8, 8))
bias2 = torch.nn.Parameter(torch.randn(1, 512, 8, 8))
