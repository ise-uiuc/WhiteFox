
class Model(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64, bias=True)
 
    def forward(self, x1):
        