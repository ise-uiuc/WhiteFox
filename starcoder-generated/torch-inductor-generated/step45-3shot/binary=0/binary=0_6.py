
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, value):
        input_tensor = x1 + 10   
        output = input_tensor + value + 0.1
        return output
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
