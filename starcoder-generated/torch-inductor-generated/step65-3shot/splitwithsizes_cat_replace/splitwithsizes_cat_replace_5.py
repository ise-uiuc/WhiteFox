
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=0)
        concatenated_tensor = torch.cat(split_tensors, dim=0)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=0))
# Inputs to the model
x1 = torch.randn(3, 2, 1)
