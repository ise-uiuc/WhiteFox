
class TestModule(nn.Module):
   def __init__(self):
       super().__init__()
   
   def forward(self, x):
       return x.view(2, 10)
 
model = TestModule()
# Inputs to the model
x = torch.randn(3, 4, 5)
