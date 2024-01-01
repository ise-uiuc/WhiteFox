
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_linear = torch.nn.Linear(100, 100)
 
    def forward(self, input):
        hidden = self.input_linear(input)
 
        alpha = 3.0
        scale = 6.0
        output = hidden * F.hardtanh(hidden, 0., 6.) + 3.0
        output = output / scale
        return output

# Initializing the model
m = Model()

# Inputs to the model,
# 2-dim input with 3 channels and sizes of height and width are 15 and 10 respectively
input = torch.randn(30, 100, 15, 10)
