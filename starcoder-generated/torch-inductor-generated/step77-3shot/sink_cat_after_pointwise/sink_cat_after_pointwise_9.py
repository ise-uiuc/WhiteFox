
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
                        torch.nn.Linear(3, 4),
                        torch.nn.Linear(4, 3),
                        torch.nn.ReLU()
                    )
    def forward(self, x):
        return self.layers(x)

# Inputs to the model
x = torch.randn(2, 3, 4)
