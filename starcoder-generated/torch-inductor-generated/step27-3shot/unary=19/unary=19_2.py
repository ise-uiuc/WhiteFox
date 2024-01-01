
class Model(torch.nn.Module):
    def __init__(self, num_in_feats, num_out_feats):
        super().__init__()
        self.linear = torch.nn.Linear(num_in_feats, num_out_feats)
 
    # Perform the forward step
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = torch.sigmoid(t1)
        return t2
 
# Initializing the model
m = Model(8, 1)
 
# Inputs to the model
x1 = torch.randn(1, 8)
