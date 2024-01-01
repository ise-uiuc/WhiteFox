
def gelu(x):
    return torch.nn.functional.gelu(x)

class Model(torch.nn.Module):
    def __init__(self, input_size=256, hidden_size=512, num_layers=2, dropout_p=0.1):
        super().__init__()

        self.decoder = torch.nn.Linear(input_size, hidden_size)

        self.decoder_layers = torch.nn.ModuleList([torch.nn.GRUCell(hidden_size, hidden_size)] * (num_layers-1))
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, X1, X2):
        d = self.decoder(X1)

        for cell in self.decoder_layers:
            d = self.dropout(d)

            d = gelu(cell(d, X2))

        d = self.dropout(d)

        d = self.decoder(d)

        d = self.decoder(d)

        return d

X1 = torch.randn(1, 256)   # Input 1 - shape of [batch_size, hidden_size]

X2 = torch.randn(1, 256)   # Input 2 - shape of [batch_size]. Usually, the input_size is the hidden size of the Decoder

