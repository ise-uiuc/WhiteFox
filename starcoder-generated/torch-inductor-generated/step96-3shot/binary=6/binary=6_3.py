
class Model(torch.nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.linear = nn.Linear(w, b, bias=True)
 
    def forward(self, x):
        v1 = linear(x)
        v2 = v1 - other
        return v2

# Initializing the model
w = 256
b = 512
m = Model(w, b)

# Inputs to the model
x = torch.randn(batch_size, input_channels, input_size, input_size)
