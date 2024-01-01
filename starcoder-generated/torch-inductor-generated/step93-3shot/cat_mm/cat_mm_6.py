
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        return torch.cat([v2, v2, v2, v2, v2, v2, v2, v2, v2], -1)
# Inputs to the model 
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
# Model begins
print('\n\nmodel1')
model1 = Model()
print(model1(x1, x2).shape)

# Input tensor:
input_size = 5
input = torch.FloatTensor(1, input_size, input_size).uniform_(0.0, 1.0)
