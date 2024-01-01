
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, other=None):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model  
__seed_weight__(m.linear, 'linear.weight', 'nn.Linear', bias=False)
__dump_input_tensor__('other')
