
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super().__init__()
        self.linear_1 = torch.nn.Linear(input_size, hidden_size)
        self.linear_2 = torch.nn.Linear(hidden_size, out_size)
 
    def forward(self, x1):
        v1 = self.linear_1(x1)
        v2 = v1 - other
        v3 = relu(v2)
        return v3

# Initializing the model
m = Model(input_size, hidden_size, out_size)

# Inputs to the model
x1 = torch.randn(sample_batch_size, input_size)
