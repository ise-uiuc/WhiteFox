
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = torch.nn.Parameter(torch.tensor([0.8]))
        self.dropout_p = torch.nn.Parameter(torch.tensor([0.7]))
        self.query = torch.nn.Parameter(torch.tensor(...))
        self.key = torch.nn.Parameter(torch.tensor(...))
        self.value = torch.nn.Parameter(torch.tensor(...))

    def forward(self, input):
        v1 = torch.matmul(self.query, self.key.transpose(-2, -1))
        v2 = v1 * self.scale_factor
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, self.dropout_p)
        v5 = torch.matmul(v4, self.value)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 30, 1024)
key = torch.randn(1, 30, 1024)
value = torch.randn(1, 30, 1024)
