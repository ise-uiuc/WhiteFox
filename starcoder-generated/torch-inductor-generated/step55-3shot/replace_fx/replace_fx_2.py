
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = torch.tensor([1,2,3,4,5], device=(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    def forward(self, x):
        x *= (self.x + self.x + self.x + self.x + self.x + self.x + self.x)
        return x
# Inputs to the model
x1 = torch.randn(1)
