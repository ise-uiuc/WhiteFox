
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 6)
 
    def forward(self, input_tensor):
        output_tensor1 = self.linear(input_tensor)
        output_tensor2 = torch.sigmoid(output_tensor1)
        output_tensor3 = output_tensor1 * output_tensor2
        return output_tensor3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(20, 4)
__o__ = m(x1)

