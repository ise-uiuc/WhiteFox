
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features1 = torch.nn.Linear(6, 8)
        self.features2 = torch.nn.Linear(6,8)
    def forward(self, x1):
        split_tensors = torch.split(x1, [2, 2, 2], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(x1, [2, 2, 2], dim=1))

# Inputs to the model
x1 = torch.randn(1, 6)
