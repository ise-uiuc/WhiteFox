
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Linear(3, 32)
    def forward(self, v1):
        concatenated_tensor = torch.cat(torch.split(v1, torch.tensor([1, 2]), dim=1), dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 2], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
