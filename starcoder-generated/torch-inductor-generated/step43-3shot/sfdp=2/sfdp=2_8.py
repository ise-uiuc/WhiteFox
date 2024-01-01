
class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.scale = input_dim ** -0.5
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.fc = torch.nn.Linear(input_dim + output_dim, output_dim)
 
    def forward(self, query, value, key):
        