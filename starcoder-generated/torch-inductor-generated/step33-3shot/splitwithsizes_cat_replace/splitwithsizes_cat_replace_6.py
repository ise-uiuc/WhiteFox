
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(3, 64, 3, 1, 1), torch.nn.AvgPool2d(3, 1, 2), torch.nn.Flatten(start_dim=1,end_dim=-1)])
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor,)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
