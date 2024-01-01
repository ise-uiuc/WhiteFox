
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(3, 8, 3, 1, bias=False), torch.nn.Conv2d(8, 8, 3, 2, bias=False)])
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        concatSplit_tensors = torch.split(concatenated_tensor, 2, dim=1)
        splitConcat_tensors = torch.split(v1, 2, dim=1)
        return(split_tensors, concatenated_tensor, concatSplit_tensors, splitConcat_tensors)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
