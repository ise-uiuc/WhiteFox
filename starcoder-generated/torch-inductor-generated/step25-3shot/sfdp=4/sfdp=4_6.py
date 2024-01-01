
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, x1, x2):
        out = torch.bmm(x1, torch.transpose(x2, 2, 1))
        mask = x1.sum(-1) # Add the attention mask to the scaled dot product
        weights = torch.softmax(mask, dim=-1) # Apply softmax to the scaled dot product
        return torch.bmm(weights, out)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 16, 3, 16)
x2 = torch.randn(5, 3, 16, 16)
