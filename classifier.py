import torch
from torch import nn
import util
import torch.distributions as dist

class KomninosManandhar(nn.Module):
    def __init__ (self, word_vocab, context_vocab, ctx_emb=300, wrd_emb=300, output_emb=300, hidden_dim=150, output_dim=-1):
        # NOTE: output_dim is set to -1 to ensure the user has passed the labels
        super().__init__()

        USE_GLOVE_EMBEDDINGS = True
        TRAIN_WORD_EMBEDDINGS = False

        # From Komninos/Manandhar (2016): "The network's weight matrices have
        # different shapes, where representations coming from the embedding
        # layer weights correspond to word embeddings, while representations
        # coming from the prediction layer weights to dependency context
        # embeddings."

        self.word_vocab = word_vocab
        self.context_vocab = context_vocab

        # We need to retain the hidden size so that we can generate initial
        # hidden states for the LSTM cell
        self.hidden_dim = hidden_dim

        self.ctx_embeddings = nn.Embedding(len(context_vocab), ctx_emb, padding_idx=self.context_vocab['<pad>'])
        self.wrd_embeddings = nn.Embedding(len(word_vocab), wrd_emb, padding_idx=self.word_vocab['<pad>'])

        # Additionally to the context and word embeddings, we need to embed the
        # output dimension. Whatever the user of this classifier wants to output,
        # we want to give them the ability to use the predictions of one sentence
        # to influence the predictions of the next sentence. Hence, we will preset
        # the initial hidden state of the LSTM layer with embeddings that are
        # trained alongside the rest of the network, and run those embeddings
        # through a form of "reverse linear layer" to get from predictions (a.k.a.
        # the results we want) to a hidden state, before we then go back from a
        # hidden state produced by the LSTM to predictions.
        # NOTE: To disable this functionality, make sure to set all predictions
        # to the padding index of one element above the output dimension size.
        # So if you're predicting three classes (male/female/unknown), the
        # padding index is 3 (=output_dim).
        self.output_embeddings = nn.Embedding(output_dim + 1, output_emb, padding_idx=output_dim)

        # Initialise all parameters using N(0, 0.1)
        distribution = dist.normal.Normal(torch.tensor([0.0]), torch.tensor([0.1]))
        matrix_size = torch.Size([len(context_vocab), ctx_emb])
        self.ctx_embeddings.weight = nn.Parameter(distribution.sample(matrix_size).squeeze(-1), requires_grad=True)
        matrix_size = torch.Size([len(word_vocab), wrd_emb])
        self.wrd_embeddings.weight = nn.Parameter(distribution.sample(matrix_size).squeeze(-1), requires_grad=TRAIN_WORD_EMBEDDINGS)
        matrix_size = torch.Size([output_dim + 1, output_emb])
        self.output_embeddings.weight = nn.Parameter(distribution.sample(matrix_size).squeeze(-1), requires_grad=True)

        # The words will be embedded using GloVe embeddings (where possible, not
        # all words are available -- the other ones will be initialised using
        # the distribution).
        if USE_GLOVE_EMBEDDINGS:
            with torch.no_grad(): # We have to freeze the weights to allow for modifications of the variables
                with open("cc.de.300.vec", "r") as fp:
                    for glove in fp:
                        embedding = glove.split()
                        if embedding[0] in word_vocab:
                            # We have a GloVe embedding!
                            idx = word_vocab[embedding[0]]
                            # BUG: I didn't see this before, but think about
                            # what we are exactly copying here. We know that
                            # embedding[0] is the word itself. Buuuut we're using
                            # the very same indexing here as well, so we're only
                            # copying 299 of the 300 weights, which means that our
                            # results are analytically completely unusable :)
                            for weight in range(len(embedding) - 1):
                                # self.wrd_embeddings.weight[idx][0] will remain initialized in the default dim
                                self.wrd_embeddings.weight[idx][weight] = float(embedding[weight + 1])
                        else:
                            # No GloVe embedding, so use the N(0,0.1) distribution samples
                            # which we applied already, so we simply don't overwrite them
                            pass

        # From Komninos/Manandhar 2016 (p. 1495 f): We need a concatenation of
        # two embeddings: One context embedding and one word embedding. We need
        # to retrieve ALL contexts for a given word, average them out, and then
        # concatenate that with the word's embedding before passing it in.
        self.lstm = nn.LSTM(
            ctx_emb + wrd_emb,
            hidden_size=hidden_dim, # See Komninos/Manandhar 2016, p. 1495
            num_layers=2, # We need two layers to apply the dropout in between
            batch_first=True,
            bidirectional=False,
            dropout=0.25, # See ibid.
            bias=True
        )

        # Define the output layer
        self.Linear = nn.Linear(hidden_dim, output_dim)

        # We need a reverse layer to feed previous class detections backwards
        # through and get our initial hidden state
        self.reverse_Linear = nn.Linear(output_emb, hidden_dim)

    def featurise (self, sentence):
        # From Komninos/Manandhar 2016, p. 1495 f:
        # "We provide syntactic information to each classifier in the following
        # manner. First we parse each sentence to get a dependency graph. Each
        # node in the graph is associated with a word w having an embedding v_w
        # and a set of dependency context features d_1 ... d_C with embeddings
        # v_d_1 ... v_d_C exactly like during the dependency based skipgram
        # training process. We then create a representation X of that node using
        # different combinations of its associated word and dependency context
        # embeddings:"
        words, contexts = util.retrieve_word_contexts(sentence)

        return_list = []

        for word, context in zip(words, contexts):
            word_idx = self.word_vocab[word] if word in self.word_vocab else self.word_vocab['<unk>']
            context_idx = [self.context_vocab[ctx] if ctx in self.context_vocab else self.context_vocab['<unk>'] for ctx in context]
            context_idx.insert(0, word_idx)
            return_list.append(context_idx)

        # In short, we have to simply return the word, and then all contexts that
        # apply to that word. And, in our forward-method, we need to retrieve
        # the embeddings for the contexts and concatenate them together.
        # Return shape must be seq_length x 1 + context_length
        return return_list

    def forward (self, features):
        # From Komninos/Manandhar (2016): "The dependency graph's node
        # representations are used as a sequence of embeddings respecting the
        # order of the sentence to become the input for the CNN and LSTM."
        #
        # Translated into what we need: We get a set of features a.k.a. each
        # word of the sentence with its associated contexts. Since we're using
        # the concatenated representation, we first need the word embedding,
        # then retrieve all context embeddings, we mean the context embeddings
        # and then concatenate the word embedding with the mean context embedding.
        batch_size, seq_length, feat_len = features.shape

        words = features[:, :, 0].unsqueeze(2) # First feature is the word
        contexts = features[:, :, 1:] # All others are contexts

        # Retrieve the corresponding embeddings.
        word_embeddings = self.wrd_embeddings(words).squeeze(2)
        context_embeddings = self.ctx_embeddings(contexts)

        # Now we have to average out all context embeddings to end up with ONE
        # context embedding per word
        context_means = torch.mean(context_embeddings[:, :, :], dim=2)

        # Concatenate both embeddings (see Komninos/Manandhar 2016, p. 1496)
        lstm_input = torch.cat((word_embeddings, context_means), dim=2)

        lstm_out = self.lstm(lstm_input)

        # Because we have two layers, we get a tuple of outputs and hiddens,
        # so we need to extract the correct one (the latter)
        outputs, (h_n, c_n) = lstm_out[1]

        linear_out = self.Linear(h_n)
        return linear_out
