
class Model(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
 
    def forward(self, query, key, value, dropout_p=0.1, inv_scale_factor=None):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        