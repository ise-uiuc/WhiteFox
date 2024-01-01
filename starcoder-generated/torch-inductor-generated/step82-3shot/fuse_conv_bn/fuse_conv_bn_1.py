
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.layer1 = torch.nn.Sequential(torch.nn.Conv1d(64, 36, 3), torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(50, 50), torch.nn.ReLU(), torch.nn.Conv2d(50, 500, 2))
    def forward(self, x1):
        x1 = self.layer1(x1)
        x1 = x1.flatten(start_dim=1)
        x1 = self.layer2(x1.unsqueeze(2)).squeeze(2)
        return x1
# Inputs to the model
x1 = torch.randn(64, 64, 2)
