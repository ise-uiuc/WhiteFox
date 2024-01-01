
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features1 = torch.nn.Sequential(*(torch.nn.MaxPool2d(5, 1, 2) for _ in range(3)))
        self.features2 = torch.nn.Sequential()
    def forward(self, x):
        split_tensors = torch.split(x, [1, 1, 1], dim=1) # Note: The split is done in the forward function, before the concat operation
        split3_tensor = split_tensors[2]
        split3_tensor_mean = torch.mean(split3_tensor, dim=[0,1,2,3])  # Note: The shape operation on the split tensor is done before the mean operation
        return (split3_tensor_mean,)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
