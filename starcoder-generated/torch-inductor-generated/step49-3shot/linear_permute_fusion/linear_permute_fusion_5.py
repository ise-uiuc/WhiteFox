
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 2)
        self.linear2 = torch.nn.Linear(2, 3)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v1, self.linear2.weight, self.linear2.bias)
        relu1 = torch.nn.functional.relu
        v4 = v3.view(x1.size(0), -1)
        v5, v6 = relu1(v4) # This is the permute function
        v7 = v5.view(1, -1)
        v8 = v5.unsqueeze(-1)
        v9 = torch.sigmoid(v7)
        v10 = v8 * v9
        return v10
# Inputs to the model
x0 = torch.randn(1, 2, 3)
