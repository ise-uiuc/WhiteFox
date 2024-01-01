
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, v1):
        split_tensors = torch.split(v1, [10, 20, 30, 40, 50], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [10, 20, 30, 40, 50], dim=1))
# Inputs to the model
x1 = torch.randn(1, 256, 64, 64)
