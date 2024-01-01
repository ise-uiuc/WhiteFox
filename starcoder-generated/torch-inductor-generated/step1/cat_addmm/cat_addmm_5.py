
from torch.nn.modules.utils import _pair

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(300, 8)

    def forward(self, x):
        out = self.fc(x)
        out = torch.cat(out)
        return out.unsqueeze(1)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
