
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_linear_layer = torch.nn.Linear(384, 8, bias=False)
        self.hidden_linear_layer = torch.nn.Linear(8, 100, bias=False)
        self.output_linear_layer = torch.nn.Linear(100, 8, bias=False)
    def forward(self, x1):
        v1 = self.input_linear_layer(x1)
        v2 = self.hidden_linear_layer(v1)
        v3 = self.output_linear_layer(v2)
        v4 = v3 + 1
        v5 = F.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 384, 19, 19)
