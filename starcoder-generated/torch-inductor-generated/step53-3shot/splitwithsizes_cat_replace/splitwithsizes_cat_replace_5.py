
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1), bias=True)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=0)
        concatenated_tensor = torch.cat(split_tensors, dim=2)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=0))
# Inputs to the model
x1 = torch.randn(3, 3, 64, 64)
