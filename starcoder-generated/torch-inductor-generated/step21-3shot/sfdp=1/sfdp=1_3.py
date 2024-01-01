
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, key, value, inv_scale_factor, p):
        