
# PyTorch does not support the syntax for multiple consecutive input tensors, so we will implement them as a sequence
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        intermediate = [torch.mm(input1, input1), torch.mm(input1, input2), torch.mm(input1, input1)] # Multiply input1 with itself; multiply input1 with input2; multiply input1 with itself again
        return intermediate[0] + intermediate[1] + intermediate[2]
# Inputs to the model:
input1 = torch.randn(32, 32)
input2 = torch.randn(32, 32)
