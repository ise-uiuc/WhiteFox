
class Layer(torch.nn.Module):
    def __init__(self, inp, hidden, out):
        super().__init__()
        self.linear1 = torch.nn.Linear(inp, hidden)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        return (torch.cat(split_tensors, dim=1), torch.cat(split_tensors))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(Layer(3, 16, 32), torch.nn.ReLU(), Layer(129, 16, 32), torch.nn.ReLU(), Layer(256, 16, 32))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
