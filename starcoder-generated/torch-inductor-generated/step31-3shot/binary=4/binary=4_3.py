
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 64, bias=True)  # 32-dimensional input, 64-dimensional output. You can add bia
    
    def forward(self, x1):
        v1 = self.linear(x1)
        