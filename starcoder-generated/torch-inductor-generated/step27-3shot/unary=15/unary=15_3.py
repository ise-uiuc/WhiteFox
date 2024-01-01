
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
        )
    
    def forward(self, x1):
        self.module.eval()
        v1 = self.module(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
