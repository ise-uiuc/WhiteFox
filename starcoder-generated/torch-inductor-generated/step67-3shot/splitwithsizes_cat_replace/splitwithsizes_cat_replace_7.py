
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 2, bias=False)
    def forward(self, v1):
        concatenated_tensor = self.fc(v1)
        return (concatenated_tensor, torch.split(concatenated_tensor, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3)
