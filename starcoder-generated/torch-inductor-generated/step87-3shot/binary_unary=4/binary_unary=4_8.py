
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(896, 1000)
 
    def forward(self, x1, x2=None, **kwargs):
        if x2 is None:
            x2 = torch.empty([]) # Initialize a dummy tensor if `x2` is not specified.
        return relu(self.linear(x1) + x2)

# Initializing the model and getting an input for inference
m = Model()
x1 = torch.randn(1, 896)
x2 = torch.randn(1, 1000)
