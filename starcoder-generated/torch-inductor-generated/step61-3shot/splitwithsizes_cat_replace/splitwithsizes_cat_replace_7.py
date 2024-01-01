
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False), torch.nn.ReLU(inplace=False)])
        self.other_features = torch.nn.Sequential(*torch.nn.ModuleList([torch.cat([torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False)], dim=1)]))
        self.cat_features = torch.nn.Sequential(*torch.cat([torch.nn.MaxPool2d(3, 2, 1) for i in range(5)], dim=1))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
