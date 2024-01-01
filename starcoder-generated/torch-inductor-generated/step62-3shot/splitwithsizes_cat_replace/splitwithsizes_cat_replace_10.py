
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, 1, 0, bias=False)
        self.maxpool = torch.nn.MaxPool2d(3, 2, 1)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        split_tensors_2nd = torch.split(concatenated_tensor, [2, 2], dim=2)
        concatenated_tensor_2nd = torch.cat(split_tensors_2nd, dim=2)
        return (concatenated_tensor_2nd, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
