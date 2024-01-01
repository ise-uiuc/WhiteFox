
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(5, 6)
        self.linear2 = torch.nn.Linear(6, 7)
     
    def forward(self, x1, x2):
        x1 = torch.softmax(x1) # Apply softmax to the input tensor x1
        x2 = self.linear1(x2) # Apply linear1 to x2
        x3 = x1 * x2 # Multiply the output of softmax with the output of linear1
        x4 = torch.matmul(x2, x1.transpose(-2, -1)) # Multiply the output of the linear1 and the transpose of the output of softmax 
        x5 = self.linear2(x4) # Apply linear2 to x4
        return x5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 5)
x2 = torch.randn(4, 6)
