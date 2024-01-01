
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(3, 8, 3, 2, 1, bias=True)])
        self.features2 = torch.nn.ModuleList([torch.nn.Conv2d(4, 4, 3, 2, 1, bias=True)])
        self.features3 = torch.nn.ModuleList([torch.nn.Conv2d(4, 4, 3, 2, 1, bias=True)])
        self.features4 = torch.nn.ModuleList([torch.nn.Conv2d(8, 8, 3, 2, 1, bias=True)])
        self.features5 = torch.nn.ModuleList([torch.nn.Conv2d(8, 8, 3, 2, 1, bias=True)])
    def forward(self, v1):
        split_tensors = torch.split(v1, (1, 1, 1), dim=-1)
        concatenated_tensor = torch.cat(split_tensors, dim=-1)
        return (concatenated_tensor, torch.split(v1, (1, 1, 1), dim=-1))
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
