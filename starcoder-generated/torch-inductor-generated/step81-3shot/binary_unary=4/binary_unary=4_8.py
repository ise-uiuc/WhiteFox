
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the inputs to the model
x1 = torch.randn(5, 10)
other = torch.tensor([[-0.2607785964487074, 0.8593767033801544, -0.48754648824567544, 0.6201909916965528, -0.7612600559294882]])
