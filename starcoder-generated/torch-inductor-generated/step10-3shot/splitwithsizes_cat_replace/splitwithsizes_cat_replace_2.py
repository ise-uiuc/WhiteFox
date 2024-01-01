
class Model(torch.nn.Module):
    def __init__(self):
        self.fc = torch.nn.Linear(3, 3)
    def forward(self, x1):
        x1 = self.fc(x1)
        split_tensors = torch.split(x1, [-1, 1], 1)
        concatenated_tensor = torch.cat(split_tensors, 1)
        return (concatenated_tensor, torch.split(x1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 3)
