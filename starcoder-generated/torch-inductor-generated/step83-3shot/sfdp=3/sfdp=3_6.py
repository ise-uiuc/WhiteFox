
class Model(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout_p, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p
        self.scale_factor = scale_factor
        self.query = torch.nn.Linear(in_features, out_features)
        self.key = torch.nn.Linear(out_features, out_features)
        self.value = torch.nn.Linear(out_features, out_features)
 
    def forward(self, q, k, v, mask):
        