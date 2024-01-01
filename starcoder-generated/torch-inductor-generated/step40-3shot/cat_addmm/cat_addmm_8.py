
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 64)
        self.relu = torch.nn.functional.relu6
    def forward(self, x):
        x = self.relu(self.layers(x))
        x = F.softmax(x)
        x = torch.flatten(x, 1)
        return x
# Inputs to the model
x = torch.randn(3, 2)
