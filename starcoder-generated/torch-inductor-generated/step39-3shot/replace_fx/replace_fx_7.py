
class SimpleRNNModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleRNNModel, self).__init__()
        self.emb_layer = torch.nn.Embedding(vocab_size, embedding_dim)
        self.rnn_layer = torch.nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, nonlinearity='relu')
        self.linear_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.softmax = torch.nn.Softmax(dim=1)
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_dim)
    def forward(self, x):
        emb = self.emb_layer(x.long()) # convert index tensor to tensor of word embeddings
        v = self.rnn_layer(emb)  # calculate LSTM output
        v = self.linear_layer(v) # apply linear transformation on LSTM output
        return self.softmax(v)
# Inputs to the model
x1 = torch.tensor([[1, 15, 2]])
