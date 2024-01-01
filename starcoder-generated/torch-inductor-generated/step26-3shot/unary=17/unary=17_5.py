
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_out = torch.nn.MaxPool2d(kernel_size=4)
    def forward(self, x1):
        v1 = self.max_out(x1) 
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
