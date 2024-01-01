
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.MaxPool2d(2, 2, 0), torch.nn.AvgPool2d(3, 2, 2, ceil_mode=False), torch.nn.MaxPool2d(3, 2, 1))
    def forward(self, v1):
        split_tensors_1 = torch.split(v1, [1, 1], dim=1)
        concatenated_tensor_1 = torch.cat(split_tensors_1, dim=1)
        split_tensors_2 = torch.split(concatenated_tensor_1, [1, 1], dim=1)
        concatenated_tensor_2 = torch.cat(split_tensors_2, dim=1)
        return (concatenated_tensor_2, torch.split(v1, [1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
