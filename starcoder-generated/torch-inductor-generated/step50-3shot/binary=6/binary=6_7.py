
class Other_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(640,136)
 
    def forward(self, x1):
        x2 = x1.reshape(1,1360)
        v1 = self.linear(x2)
        return v1
        
# Initializing the model
a = Other_model()

# Inputs to the model
__other__ = torch.randn(1, 3, 64, 64)
v2 = __other__.reshape(1,640)
__input__ = v2
