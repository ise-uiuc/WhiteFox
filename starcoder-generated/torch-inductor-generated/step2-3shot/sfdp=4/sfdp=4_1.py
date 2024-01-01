
class Model(torch.nn.Module):
    def __init__(self, config, input_shape):
        super().__init__()
        self.proj = torch.nn.Linear(input_shape, config.hidden_size)
 
    def forward(self, x):
        x = self.proj(x)
        return x

# Initializing the model
config = CONFIG["attention"]()
input_shape = (1, 4, 20)
m = Model(config, input_shape)

# Inputs to the model
x = torch.randn(input_shape)
