
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(784, 200),
            torch.nn.Tanh(),
            torch.nn.Linear(200, 20),
            torch.nn.Tanh(),
            torch.nn.Linear(20, 10))
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.model(x)
# Inputs to the model
tensor = torch.randn(1, 1, 28, 28)
