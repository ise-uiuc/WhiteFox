
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
    def forward(self, input_1):
        output = torch.nn.functional.linear(input_1, self.linear.weight, self.linear.bias)
        output_2 = output.permute(0, 2, 1)
        return output_2.permute(0, 2, 1)
# Inputs to the model
input_1 = torch.randn(2, 3)
