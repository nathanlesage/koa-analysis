# This is the parser module that is able to parse annotated data files.
from collections import Counter

def parse_preprocessed (filename):
  """
  Parses a preprocessed (not raw!) file into a Python data structure.
  """
  with open(filename, 'r', encoding='utf-8') as file:
    sentence = []
    for idx, line in enumerate(file):
      sanitized = line.strip()
      # Also check the length of a sentence to make sure we don't have a double
      # newline, which happens at the end of a file.
      if sanitized == '' and len(sentence) > 0:
        # Empty line means end of sentence
        yield sentence
        sentence = []
        continue

      if sanitized == '':
        # Occasionally, there may be multiple newlines in the file.
        continue

      # Extract the information
      token_id, text, head, upos, deprel, label = sanitized.split("\t")
      # A token ID of 1 indicates next sentence
      if int(token_id) == 1 and len(sentence) > 0:
        yield sentence
        sentence = []

      sentence.append({
        'id': int(token_id),
        'text': text,
        'head': int(head),
        'upos': upos,
        'deprel': deprel,
        'label': label
      })

    if len(sentence) > 0:
      yield sentence # One final yield


def make_vocabs (datafile):
    """
    Generates word- and context-vocabularies given in the datafile.
    """
    # From Levy/Goldberg 2014: "All tokens were converted to lowercase, and words
    # and contexts that appeared less than 100 times were filtered."

    word_freq = Counter()
    context_freq = Counter()
    for sentence in parse_preprocessed(datafile):
        # Retrieve the words and contexts, and update our counters
        words, contexts = retrieve_word_contexts(sentence)
        word_freq.update(set(words))
        all_contexts = []
        for context in contexts:
            all_contexts += context # Concat

        context_freq.update(set(all_contexts))

    # Now throw out everything that appears less than 100 times and assign
    # correct indices.
    word_vocab = {
        '<pad>': 0,
        '<unk>': 1
    }
    contexts_vocab = {
        '<pad>': 0, # We need to pad the contexts to uniform length
        '<unk>': 1
    }

    for k, c in word_freq.items():
        if c >= 1:
            word_vocab[k] = len(word_vocab)

    for k, c in context_freq.items():
        if c >= 1:
            contexts_vocab[k] = len(contexts_vocab)

    return word_vocab, contexts_vocab

def retrieve_word_contexts (sentence):
    """
    Retrieves all contexts for all words in the sentence.
    """
    words = []
    contexts = []
    for token in sentence:
        # With respect to Figure 1 in Levy/Goldberg 2014, we have to extract
        # all possible contexts for the current word/token.
        words.append(token['text'])
        internal_contexts = []
        if token['head'] > 0:
            # TODO: We have to "collapse" multiple dependencies, so we have
            # to follow ALL dependency relations to the root
            head = sentence[token['head'] - 1]
            internal_contexts.append(f"{head['text']}/{token['deprel']}-1")

        # Retrieve all dependants for this token and add them
        dependants = [dep for dep in sentence if dep['head'] == token['id']]
        for dep in dependants:
            internal_contexts.append(f"{dep['text']}/{dep['deprel']}")

        contexts.append(internal_contexts)

    return words, contexts # int, (int, num_contexts)
