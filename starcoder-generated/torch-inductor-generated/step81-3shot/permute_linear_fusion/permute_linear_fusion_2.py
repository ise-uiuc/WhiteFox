
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.flatten = torch.nn.Flatten(0, 1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        # TODO: Implement this pattern by invoking the relu and the max function with dim arg equal to -1. Then use it to index and compute the sum, and get the unbiased tensor by addition. Finally, permute the result to swap the last two dimensions.
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
