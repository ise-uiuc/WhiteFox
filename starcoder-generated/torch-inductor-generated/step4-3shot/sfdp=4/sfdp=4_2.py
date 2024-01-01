
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, __input__):
        q = __input__[:, 0]
        k = __input__[:, 1]
        v = __input__[:, 2]
        