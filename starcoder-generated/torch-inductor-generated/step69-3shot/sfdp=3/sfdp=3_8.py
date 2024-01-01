
class Model(torch.nn.Module):
    def __init__(self, input_size, batch_size, seq_len, num_head, hidden_size, num_layer, dropout_p, seed=0):
        super().__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.dropout_p = dropout_p

        self.seed = seed
        torch.manual_seed(seed)

        self.embed = torch.nn.Linear(input_size, hidden_size) 

        self.attentions = torch.nn.ModuleList()
        for _ in range(num_layer):
            new_attention = Attention(hidden_size, num_head)
            self.attentions.append(new_attention)

        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x1, x2):
        x1 = self.embed(x1)
        x2 = self.embed(x2)

        v = torch.zeros(seq_len, hidden_size)
        for i in range(num_layer):
            v = self.attentions[i](x1, x2, v)
            v = self.dropout(v)

        return v

# Initializing the model
m = Model(input_size, batch_size, seq_len, num_head, hidden_size, num_layer, dropout_p, seed)

# Inputs to the model
x1 = torch.randn(batch_size, input_size)
x2 = torch.randn(batch_size, input_size)
