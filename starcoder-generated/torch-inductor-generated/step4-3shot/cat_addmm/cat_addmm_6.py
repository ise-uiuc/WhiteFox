
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
        )
 
    def forward(self, x1, x2):
        v1 = self.fc(x2)
        v2 = torch.addmm(x1, v1, v1.t())
        v3 = torch.cat((v2, x1, torch.zeros_like(x1)), dim=1)
        return v3
 