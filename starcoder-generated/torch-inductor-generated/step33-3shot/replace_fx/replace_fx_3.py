
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self, output_dim):
        super(Model, self).__init__()
        self.dense1 = torch.nn.Linear(2, output_dim)
        
        for n, p in self.dense1.named_parameters():
            if 'weight' in n:
                torch.nn.init.normal_(p, std=1.0)
                
    def forward(self, X, W):
        x1 = F.normalize(X, p=2, dim=1)
        d1 = self.dense(x1)
        return d1
# Inputs to the model
X = torch.randn(1, 2, requires_grad=True)
W = torch.empty(1, requires_grad=False)
