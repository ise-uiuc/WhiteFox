
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
        self.k = torch.randn(8, 64, 1024)
        self.k_t = self.k.transpose(-2, -1)
        self.inv_scale_factor = 1
        self.p = 0.4
        self.softmax = torch.nn.Softmax(-1)
        self.dropout = torch.nn.Dropout(p=self.p)
        self.v = torch.randn(8, 8, 1024)
 
    def forward(self, x):
        queryv1 = torch.matmul(x, self.k_t)
        queryv2 = queryv1 / self.inv_scale_factor
        queryv3 = self.softmax(queryv2)
        queryv4 = self.dropout(queryv3)
        queryv5 = torch.matmul(queryv4, self.v)
        return queryv5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 1024)
