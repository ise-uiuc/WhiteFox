
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        add = False
        if x1.size()[1]!= 16:
            add = True
            
        v1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)(x1)
        v2 = v1 + x2
        v3 = torch.nn.functional.relu(v2)
        v4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)(v3)
        v5 = v4 + v3
        v6 = torch.nn.functional.relu(v5)
        v7 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
