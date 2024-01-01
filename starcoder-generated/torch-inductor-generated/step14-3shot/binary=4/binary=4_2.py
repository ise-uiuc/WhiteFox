
class TorchLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(in_features=3, out_features=1, requires_grad=True))
 
    def forward(self, x):
        y = torch.nn.functional.linear(x, self.weight)
        z = y + x
        return z

# Initializing the model
m = TorchLinearModel()

# Inputs to the model
x = torch.randn(1, 3)
