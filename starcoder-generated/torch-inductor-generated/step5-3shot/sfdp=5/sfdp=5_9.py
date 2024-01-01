
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = torch.nn.Transformer(d_model=512)
        self.model = model
        self.linear = torch.nn.Linear(512, 128)

    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = self.model(v1, x2)
        v3 = self.linear(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10,128)
x2 = torch.randn(10, 10, 512)
