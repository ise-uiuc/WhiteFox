
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, inv_scale_factor, dropout_p=0.5, value=None):
        output = 0
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 4, 512)
key = torch.randn(1, 4, 512)
inv_scale_factor = torch.randn(1, 512)
