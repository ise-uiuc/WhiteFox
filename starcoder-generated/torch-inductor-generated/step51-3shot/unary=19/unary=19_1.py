
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, hidden_size)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model with specified parameters
_ = torch.manual_seed(1024)
input_size = 60
hidden_size = 70
output_size = 7
m = Model(input_size, hidden_size, output_size)

# Inputs to the model
x1 = torch.randn(1, 60)
