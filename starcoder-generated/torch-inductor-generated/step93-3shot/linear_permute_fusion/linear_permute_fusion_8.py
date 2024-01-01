
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1, x2):
        v5 = x1
        v6 = x2
        v7 = self.linear1(v5)
        v8 = self.linear2(v6)
        v1 = v7.permute(0, 2, 1) + v8.permute(0, 2, 1) # Apply linear transformation to the input tensor.
        v2 = v1.permute(0, 2, 1)                             # Permute the output tensor from the linear transformation.
        v3 = self.linear1(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
