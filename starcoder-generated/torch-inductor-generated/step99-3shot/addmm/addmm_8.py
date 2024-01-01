
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x1 = torch.randn(3, 3)
        self.x2 = torch.randn(3, 3)
    def forward(self, y1, y2):
        v1 = torch.mm(y1, self.x1)
        v1 = torch.mm(v1, self.x2)

        v2 = torch.mm(y2, self.x1)
        v2 = torch.mm(v2, self.x2)
    
        return v1 + v2
# Inputs to the model
y1 = torch.randn(3, 3)
y2 = torch.randn(3, 3)
