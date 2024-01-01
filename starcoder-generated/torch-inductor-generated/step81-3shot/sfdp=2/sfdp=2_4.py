
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torch.nn.Conv2d(3, 8, 1)
 
    def forward(self, X=None, K=None, V=None):
        q = self.m(X)
        k = self.m(K)
        v = self.m(V)
        scale_factor = q.shape[2] * q.shape[3] 