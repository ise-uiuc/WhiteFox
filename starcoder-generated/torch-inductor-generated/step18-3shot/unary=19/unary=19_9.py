
class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
 
    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

# Initializing the model
m = Model(20, 10)

# Input to the model
x = torch.randn(10, 20)
