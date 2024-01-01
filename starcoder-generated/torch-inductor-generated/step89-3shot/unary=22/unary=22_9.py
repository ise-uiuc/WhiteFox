
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.linear_1 = torch.nn.Linear(input_size, hidden_size)
        self.linear_2 = torch.nn.Linear(hidden_size, num_classes)
 
    def forward(self, x):
        v1 = torch.tanh(self.linear_1(x))
        return self.linear_2(v1)

# Initializing the model
input_size = 10
hidden_size = 10
num_classes = 10
m = Model(input_size, hidden_size, num_classes)

# Inputs to the model
x = torch.randn(1, 10)
