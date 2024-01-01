
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
 
    def forward(self, input_tensor, other=10):
        output1 = self.linear(input_tensor)
        output2 = output1 - other
        return output2
# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(1, 1)
