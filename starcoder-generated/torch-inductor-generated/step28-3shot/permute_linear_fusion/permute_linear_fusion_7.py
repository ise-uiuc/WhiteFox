
# A linear operation on a permuted tensor is equivalent to a transformation on a reshaped tensor. 
# So you can replace the permute methods with rehapes so the pattern is easier to trigger using reshapes API.

# The reshaped tensor will be 4D tensor (1, 1, W * H, 2), because the original tensor (1, 2, W, H).
# So you can replace the permute methods used on the original tensor with the reshape() method as such:
# tensor.reshape((1, 1, 2, input_tensor.size(2) * input_tensor.size(3))) 
# However, this will not change the input tensor since all reshapes do not preserve the original tensor's shape and data types.

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.reshape(1, 1, 2, x1.size(2) * x1.size(3))
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2 = torch.tanh(v2)
        x2 = torch.nn.functional.threshold(v2, -0.8, 0.8, False)
        v4 = x2 * x2
        v3 = torch.nn.functional.linear(v1, self.linear.weight * 2, self.linear.bias)
        v3 = v3 + x2
        return v2 + v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
