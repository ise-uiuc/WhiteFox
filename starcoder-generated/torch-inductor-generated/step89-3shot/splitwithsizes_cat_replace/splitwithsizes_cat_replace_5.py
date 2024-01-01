
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 5, 1, 2, bias=False)
    def forward(self, _x):
        # FIXME: the split and concat dim are not the same and the concat order is not the same as the split order that can result in redundant transpose operations.
        split_tensors = torch.split(_x, [5, 62, 62], dim=3 if _x.size(3) == 64 else 2)
        concatenated_tensor = torch.cat(split_tensors, dim=3 if _x.size(3) == 64 else 2)
        flatten_tensor = flatten(concatenated_tensor, start_dim=1)
        return (concatenated_tensor, flatten_tensor)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
