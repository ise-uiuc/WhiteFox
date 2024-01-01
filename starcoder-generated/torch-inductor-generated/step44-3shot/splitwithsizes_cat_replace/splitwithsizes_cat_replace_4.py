
class Layer1(torch.nn.Module):
    def __init__(self, inp, hidden, out):
        super().__init__()
        self.linear1 = torch.nn.Linear(inp, hidden, bias=False)
        self.linear2 = torch.nn.Linear(hidden, out, bias=False)
    def forward(self, v1, v2):
        return torch.addmm(input=self.linear2(torch.relu(self.linear1(v1))), input2=torch.relu(v2), beta=1,alpha=1, mat1=self.linear1.weight.t(), mat2=self.linear1.weight)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Layer1(3, 8, 4)
        self.extra = torch.nn.ReLU()
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64)
x2 = torch.randn(1, 4)
