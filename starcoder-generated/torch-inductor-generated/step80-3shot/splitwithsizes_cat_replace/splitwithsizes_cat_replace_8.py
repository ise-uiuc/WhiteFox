
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential()
    def forward(self, v1, v3):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        split_tensors_v3 = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor_v3 = torch.cat(split_tensors_v3, dim=1)
        return (concatenated_tensor, split_tensors, split_tensors_v3, concatenated_tensor_v3)
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x3 = torch.randn(1, 3, 32, 32)
