 input
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, negative_slope):
        super().__init__()

        self.first_linear = nn.Linear(in_features=input_dim, out_features=hidden_dims[0])
        self.hidden = nn.ModuleList([nn.Linear(in_features=d, out_features=2*d) 
                                      for i, d in enumerate(hidden_dims[1:])])
        self.last_linear = nn.Linear(in_features=hidden_dims[-1], out_features=num_classes)

        self.negative_slope = negative_slope
 
    def forward(self, x1):
        