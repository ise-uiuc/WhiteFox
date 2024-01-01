
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 8)
    def forward(self, x1):
        # This model meets the requirement since all input tensors to linear operator are 4D tensor and the last dimension of
        # input tensor is the same as the last dimension of weight tensor of linear operator.
        v3 = x1
        v2 = torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias)
        return v1
# Inputs to the model
x1 = torch.randn(4, 4)
