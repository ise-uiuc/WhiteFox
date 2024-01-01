
class Model(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
 
    def forward(self, t1, t2):
        v1 = self.linear(t1)
        v2 = v1 - v2
        return v2

# Initializing the model
m = Model(10, 5)

# Inputs to the model
t1 = torch.randn(5, 10)
t2 = torch.randn(5, 10)
