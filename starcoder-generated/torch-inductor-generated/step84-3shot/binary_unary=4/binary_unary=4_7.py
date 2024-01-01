
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
 
    def forward(self, x1, cifar10_input_transform=None):
        v1 = None
        if cifar10_input_transform == '1':
            v1 = self.linear(x1 + 0.5) # apply the linear transformation to x1 and add 0.5
        else:
            v1 = self.linear(x1) # apply the linear transformation to x1
        v2 = v1 + 10 # add 10 to the output of the linear transformation
        v3 = torch.nn.functional.relu(v2) # apply the ReLU activation function to the result
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
