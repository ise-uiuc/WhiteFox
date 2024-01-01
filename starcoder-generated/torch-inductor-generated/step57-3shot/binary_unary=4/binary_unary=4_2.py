
class Model(torch.nn.Module):
    def __init__(self, w1):
        super().__init__()
        self.w1 = torch.nn.Parameter(w1)
 
    def forward(self, x1):
        v1 = torch.matmul(x1, self.w1)
        return v1 + x1

# Initializing the model
input_channels = 100
num_output_channels = 50
w1 = torch.rand((num_output_channels, input_channels))
m = Model(w1)
 
# Inputs to the model, including one keyword argument
x1 = torch.randn(10, input_channels)
