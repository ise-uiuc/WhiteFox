
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, v1):
        split_tensors = [torch.split(v1, [1, 1, 1], dim=1)] * 2
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (torch.cat(concatenated_tensor, dim=1), torch.stack(split_tensors, dim=2))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
