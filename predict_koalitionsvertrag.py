# Here we simply load the classifier and have it predict the Koalitionsprogramm
import torch
from torch.nn import Softmax
from classifier import KomninosManandhar
from tqdm import tqdm
import stanza

model_file = "model_state.torch"

# First load the data file
print(f"Loading model data from {model_file} ...")
state = torch.load(model_file)

# Now the state contains several dictionary keys we need
word_vocab = state['word_vocab']
context_vocab = state['context_vocab']
labels = state['labels']

# Recreate the model
print("Instantiating model ...")
model = KomninosManandhar(word_vocab, context_vocab, output_dim=len(labels))
model.load_state_dict(state['model'])
model.eval()

# Load stanza
nlp = stanza.Pipeline('de', processors="tokenize,pos,lemma,depparse")

# Now, predict the whole Koalitionsprogramm (which party most likely has written each sentence)
with open("raw/koalitionsvertrag_ampel_2021.txt", "r") as fp:
  koalitionsvertrag = fp.read()

koalitionsvertrag = koalitionsvertrag.replace("\n", " ")


results = list()
print("Parsing document ...")
doc = nlp(koalitionsvertrag) # This can take a few minutes
print("Done parsing!")

for idx, sentence in tqdm(enumerate(doc.sentences), desc="Predicting", unit="sentence"):
  tokens = []
  for word in sentence.words:
    tokens.append({
      'id': word.id,
      'text': word.text,
      'head': word.head,
      'upos': word.upos,
      'deprel': word.deprel
    })

  features = model.featurise(tokens)

  # Now we have to pad the features to equal length
  max_len = 0
  for feature in features:
      if len(feature) > max_len:
          max_len = len(feature)
  
  for feature in features:
      while len(feature) < max_len:
          feature.append(0)

  # Add an outer dimension (since we effectively have a batch size of 1 here)
  tensor = torch.LongTensor(features).unsqueeze(0)

  predictions = torch.squeeze(model.forward(tensor)) # Remove the outer dimension again

  # Get the highest scoring indices
  percentages = Softmax(dim=0)(predictions)
  pred_idx = torch.argmax(percentages)


  # Append to results
  results.append((idx, sentence.text, percentages[0], percentages[1], percentages[2], labels[pred_idx]))

# Write to file
with open("koalitionsvertrag_final.tsv", "w") as fp:
  fp.write(f"Index\tSentence\t{labels[0]}\t{labels[1]}\t{labels[2]}\tMost likely\n")
  for result in results:
    idx, sentence, one, two, three, label = result
    fp.write(f"{idx}\t{sentence}\t{one}\t{two}\t{three}\t{label}\n")
