
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        tensors = torch.split(x1, [2, 4, 10, 6], 1)
        x2 = torch.cat([tensors[i] for i in range(len(tensors))], 1)
        return x2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5, 10, 10)
