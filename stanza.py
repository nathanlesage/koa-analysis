# This file transforms the prepared, cleaned data files into features
import stanza
from tqdm import tqdm

# NOTE: Uncomment the following to download the German stanza data if not already done
# stanza.download('de', processors="tokenize,pos,lemma,mwt,ner,depparse")

nlp = stanza.Pipeline('de', processors="tokenize,pos,lemma,depparse")

def parse_file (filename, label=None):
  if label is None:
    raise ValueError("Label was none!")

  with open(filename, "r") as fp:
    contents = fp.read().strip() # Remove potential newlines at the end
    if contents == '':
      return

    contents = contents.replace("\n", " ") # Newlines -> spaces

    parsed_doc = nlp(contents)
    with open(f"stanza.{filename}", "w") as fp:
      for sentence in parsed_doc.sentences:
        for word in sentence.words:
          # text    Original input text
          # head    The head for this word in the sentence (position, w/ pseudo-root)
          # upos    The token's POS tag (according to Universal POS)
          # deprel  The dependency relation of that word with regard to the sentence (e.g., nsubj, det, punct, or root)
          try:
            fp.write(f"{word.id}\t{word.text}\t{word.head}\t{word.upos}\t{word.deprel}\t{label}\n")
          except:
            print(f"Error in word {word}. Skipping.")

# Now parse the files
parse_file("dehyph.fdp_btw2021.txt", label='fdp')
parse_file("dehyph.spd_btw2021.txt", label='spd')
parse_file("dehyph.gruene_btw2021.txt", label='gruene')
