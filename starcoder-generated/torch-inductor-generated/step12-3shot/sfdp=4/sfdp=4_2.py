
class Model(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
    
    def forward(self, x1, x2, x3):
        