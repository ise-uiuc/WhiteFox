
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, inv_scale_factor: float, dropout_p: float):
        pass

# Initializing the model
dropout_p = 0.8
inv_scale_factor = 1.0 / np.power(dropout_p, 0.6)
m = Model()

# Inputs to the model
query = torch.randn(512, 5, 200)
key = torch.randn(512, 5, 200)
value = torch.randn(512, 5, 300)
