
class Model(torch.nn.Module):
    def forward(self, input_tensor, other):
        output_tensor = torch.nn.functional.relu(torch.nn.functional.linear(input_tensor, other, bias=None))
        return output_tensor

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 5)
other = torch.randn(4, 5)
