
class Model(torch.nn.Module):
    def __init__(self, input_dim=5, projection_dim=10, dropout_p=0.6):
        super().__init__()
 
    def forward(self, query, key, value):
        