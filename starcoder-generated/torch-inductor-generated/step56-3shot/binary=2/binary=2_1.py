
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 2, stride=1, padding=1),
            torch.nn.Flatten()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(16 * 16, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        v1 = self.layer1(x1)
        v2 = self.layer2(v1) 
        # Concat the two inputs
        v3 = torch.cat((v2, x2), 1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 5, 32, 32)
