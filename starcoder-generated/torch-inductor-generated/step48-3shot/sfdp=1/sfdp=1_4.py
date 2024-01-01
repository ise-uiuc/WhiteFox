
emb_input = 3 # Dimensionality of inputs to the model
emb_hidden = 2 # Dimensionality of the embeddings
proj_output = 2 # Dimensionality of the output from linear projections

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embs = torch.nn.Embedding(vocab_size, emb_input)
        self.proj1 = torch.nn.Linear(emb_input, emb_hidden)
        self.proj2 = torch.nn.Linear(emb_hidden, emb_hidden)
        self.proj3 = torch.nn.Linear(emb_hidden, proj_output)
        self.dropout_in = torch.nn.Dropout(p=0.2)
        self.dropout_hidden = torch.nn.Dropout(p=0.2)
 
    def forward(self, indices):
        embs = self.embs(indices)
        hidden = self.proj1(embs)
        hidden = torch.relu(hidden)
        hidden = self.dropout_in(hidden)
        hidden = self.proj2(hidden)
        hidden = torch.tanh(hidden)
        hidden = self.dropout_hidden(hidden)
        logits = self.proj3(hidden)
        return logits

# Initializing the model
m = Model()

# Model description
m.train()

embs = torch.randint(size=(1, 128), low=0, high=vocab_size-1)
logits = m(embs)

