
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(3, 3)
 
    def forward(self, x):
        x = self.linear(x)
        x = x * torch.clamp(torch.nn.functional.linear(x, x), min=0, max=6) + 3
        x = x / 6
        return x

# Input to the model
model_input = torch.randn(1, 3)
