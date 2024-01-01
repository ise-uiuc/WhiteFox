
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, v1):
        split_tensors1 = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor1 = torch.cat(split_tensors1, dim=1)
        split_tensors2 = torch.split(concatenated_tensor1, [1, 1, 1], dim=1)
        concatenate_tensor2 = torch.cat(split_tensors2, dim=1)
        return (concatenate_tensor2, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
