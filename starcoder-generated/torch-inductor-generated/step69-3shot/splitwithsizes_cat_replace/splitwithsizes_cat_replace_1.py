
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Conv2d(3, 4, 1, 1, 0)
    def forward(self, v1):
        split_tensors = torch.split(v1, [4, 4], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return concatenated_tensor
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = Model1()
        self.branch2 = Model1()
        self.other_feature = torch.nn.Linear(8, 1)
    def forward(self, v1):
        split_tensors = torch.split(v1, [4, 4], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        split_tensors2 = torch.split(concatenated_tensor, 2, dim=1)
        concatenated_tensor2 = torch.cat(split_tensors2, dim=1)
        split_tensors3 = torch.split(concatenated_tensor2, 2, dim=1)
        concatenated_tensor3 = torch.cat(split_tensors3, dim=1)
        return (concatenated_tensor3)
# Inputs to the model
x1 = torch.randn(1, 12, 64, 64)
