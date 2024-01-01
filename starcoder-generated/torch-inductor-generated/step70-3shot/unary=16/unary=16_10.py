
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.act = torch.nn.ReLU()
 
    def forward(self, x):
        v1 = self.fc1(x)
        v2 = self.act(v1)
        return v2

# Initializing the model
m = Model(3, 100, 3)

# Inputs to the model
x = torch.randn(1, 3)
