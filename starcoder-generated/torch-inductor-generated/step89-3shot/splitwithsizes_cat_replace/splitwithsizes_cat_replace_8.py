
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Conv2d(3, 32, 3, 1, 0, bias=False), torch.nn.Identity(), torch.nn.MaxPool2d(3, 1, 2)])
        self.identity = torch.nn.Identity()
    def forward(self, x1):
        split_tensors = torch.split(x1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(x1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
