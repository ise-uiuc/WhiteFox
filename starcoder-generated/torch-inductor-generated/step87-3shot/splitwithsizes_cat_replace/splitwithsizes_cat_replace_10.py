 
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 1),
            torch.nn.LeakyReLU(negative_slope=1.0, inplace=False),
            torch.nn.ReLU(inplace=False)
        )

    def forward(self, inp, index=None):
        if index is None:
            return self.block(inp)

        return self.block(inp)[index]

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = MyModule()
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
