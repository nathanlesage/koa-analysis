# Define the training loop

# This module trains a classifier on the provided training data.
from classifier import KomninosManandhar
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import util
import datetime

# According to https://stackoverflow.com/a/66002960, a simple heuristic is
# apparently the best we can do for finding the primary subject/object of a
# sentence.
def training_examples (model, gold_data, labels, window_size=5, seq_len=25, batch_size=100):
    """
    Generates training examples based on gold-standard data.
    """
    batch_features = torch.zeros((batch_size, seq_len, window_size), dtype=torch.long)
    batch_prediction = torch.zeros(batch_size, dtype=torch.long)

    batch_idx = 0
    for sentence in util.parse_preprocessed(gold_data):
        features = model.featurise(sentence)

        # Now append to all lists
        for token_position in range(seq_len):
            if token_position == len(features):
                break
            for feat in range(len(features[token_position])):
                batch_features[batch_idx][token_position][feat] = features[token_position][feat]

        # We just take any of the labels per sentence (which are all the same)
        batch_prediction[batch_idx] = labels.index(sentence[0]['label'])

        batch_idx += 1

        if batch_idx == batch_size:
            yield batch_features, batch_prediction
            batch_idx = 0
            batch_features = torch.zeros((batch_size, seq_len, window_size), dtype=torch.long)
            batch_prediction = torch.zeros(batch_size, dtype=torch.long)

    if batch_idx > 0:
        yield batch_features[:batch_idx], batch_prediction[:batch_idx]

def train (n_epochs=10, batch_size=100, lr=1e-2, labels=[], train_data=None):
    """
    Trains a KomninosManandhar classifier on the given training data
    """
    if len(labels) == 0:
      raise ValueError("Labels must be a list of labels to find in the data!")

    print("Beginning training of the Komninos/Manandhar classifier.")

    # Create the vocabularies
    word_vocab, contexts_vocab = util.make_vocabs(train_data)
    print("")
    print(f"Using vocabulary sizes of {len(word_vocab)} words and {len(contexts_vocab)} contexts.")

    print(f"Binding classifiers to {len(labels)} labels: {', '.join(labels)}")

    # Prepare the models
    print("")
    print("Beginning instantiation of classifier. This might take a while since it will be loading embeddings ...")
    model = KomninosManandhar(word_vocab, contexts_vocab, output_dim=len(labels))
    print("Classifier instantiated.")
    print("")

    # Prepare the window size, because to speed up computation we want to train
    # fixed window sizes (this way we can employ matrix operations).
    window_size = 0
    for sentence in util.parse_preprocessed(train_data):
        features = model.featurise(sentence)
        if len(features) > window_size:
            window_size = len(features)

    print(f"Performing classification using window sizes of {window_size}")

    # Prepare the optimizers
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epoch_losses = []
    accuracies = []

    print("Initialisation sequence complete. Entering training loop ...")

    progress = tqdm(total=n_epochs, position=0)
    progress.set_description("Training")
    for epoch in range(n_epochs):
        losses = []
        running_accs = []
        run = 0
        progress.update()
        for features, true_labels in training_examples(model, train_data, labels, window_size=window_size, batch_size=batch_size):
            run += 1

            optimizer.zero_grad()
            predictions = model.forward(features)
            loss = F.cross_entropy(predictions, true_labels)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            # Update the progress bar with our new information
            progress.set_postfix({
                'loss': torch.mean(torch.tensor(losses)).item()
            })

        # After each epoch, append information about the run to our lists
        epoch_losses.append(torch.mean(torch.tensor(losses)))

    progress.close()

    # In the end, print out our information on a per-epoch basis
    print("Training finished.")
    print("")
    print("Losses:")
    print(epoch_losses)
    print("Training done.")
    # TODO: Save model to disk
    torch.save({
      'model': model.state_dict(),
      'word_vocab': model.word_vocab,
      'context_vocab': model.context_vocab,
      # Important: Since the classifiers are bound to the labels from training,
      # we have to retain these in our saved state!
      'labels': labels
    }, "model_state.torch")

if __name__ == "__main__":
  print("Training the classifier!")
  train(n_epochs=10, batch_size=100, lr=1e-2, labels=["spd", "gruene", "fdp"], train_data="train_data.txt")
