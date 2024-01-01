
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 18)
 
    def forward(self, x):
        v1_ = self.linear(x) # Linear Transformation on x
        v2 = torch.relu(v1_) # ReLU Activation function on output of linear transformation on x
        return v2
 
# Initializing model
m = Model()

# Inputs to the model
x = torch.randn(16,10)
