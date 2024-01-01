
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        output = torch.nn.functional.linear(input=input_tensor, weight=None, bias=None) - other
        return output

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2)
y = torch.randn(1, 2)
