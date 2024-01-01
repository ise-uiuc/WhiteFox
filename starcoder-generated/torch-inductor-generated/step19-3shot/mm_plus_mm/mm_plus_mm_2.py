
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x): # x (3, 16, 16)
        x = x + 1 # (3, 16, 16) tensor
        x = x[0:1,...] # (1, 16, 32) tensor
        x = x.squeeze(0) # (16, 32) tensor
        x = x.transpose(0, 1) # (32, 16) tensor
        x = torch.cat([x, x, x]) # (32, 32) tensor
        return x.sum(1) # (32,) tensor
# Inputs to the model
x = torch.randn(3, 16, 16)
