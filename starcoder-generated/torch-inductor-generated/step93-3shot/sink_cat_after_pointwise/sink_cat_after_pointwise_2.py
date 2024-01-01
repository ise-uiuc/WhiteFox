
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
                    torch.nn.ZeroPad2d((2, 3, 2, 3)),
                    torch.nn.Conv1d(1, 20, 5),
                    torch.nn.Conv1d(20, 20, 5),
                    torch.nn.ReLU(),
                    torch.nn.Conv1d(20, 20, 5)
                )

    def forward(self,x):
        return self.model(x)

# Inputs to the model
x = torch.randn(1, 1, 32)
