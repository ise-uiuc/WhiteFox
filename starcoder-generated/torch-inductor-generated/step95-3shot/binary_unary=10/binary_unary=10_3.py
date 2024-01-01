
class Model(torch.nn.Module):
    def __init__(self, w_init):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(2, 2, bias=True)
 
        # The weight is initialized with the input tensor
        self.linear.weight.data = torch.tensor(w_init, dtype=torch.float32) 
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return v1 + x
  
# Inputs to the model
x1 = torch.randn(1, 2)
w_init = np.array([
    [0, 1],
    [1, 0]])
