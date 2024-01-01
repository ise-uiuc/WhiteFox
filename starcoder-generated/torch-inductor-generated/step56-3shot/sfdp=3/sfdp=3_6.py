
class Model(torch.nn.Module):
    def __init__(self, __input_shape__):
        super().__init__()
        self.linear1 = torch.nn.Linear(n_hidden, n_hidden * 4)
        self.linear2 = torch.nn.Linear(n_hidden * 4, n_hidden)
 
    def forward(self, w, x):
        
        w1 = self.linear1(w)
        w2 = torch.nn.functional.dropout(torch.nn.functional.relu(w1), p=dropout_p)
       
        x1 = torch.cat((x, w2), -1)
        x2 = self.linear2(x1)
        x3 = x2 * torch.sqrt(torch.as_tensor(n_hidden))
        return x3

# Initializing the model
m = Model(__input_shape__)

# Inputs to the model
w = torch.randn(1, n_hidden)
x = torch.randn(1, __input_shape__)
x1 = m(w, x)

