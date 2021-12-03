# This file basically dehyphenates the Wahlprogramme

vocab = {}
def_wrong = {}
dict_de = {}

# Load dictionary
with open("dict_german.txt", "r") as fp:
  for word in fp.readlines():
    dict_de[word.strip().lower()] = True
  print(f"Loaded {len(dict_de)} dictionary words.")

# Load cached specific words
with open("vocab.txt", "r") as fp:
  for word in fp.readlines():
    vocab[word.strip()] = True
  print(f"Loaded {len(vocab)} custom words.")

def is_correct (word):
  if word.lower() in vocab or word.lower() in dict_de:
    return True
  else:
    return False

def dehyphenate (document):
  # Preprocessing: Clean up the lines
  lines = [line.strip() for line in document.split("\n")]

  for idx, line in enumerate(lines):
    if line == "" or " " not in line or line[0] == "\f":
      continue

    words = line.split(" ")
    lastword = words[-1]
    # Remove a hyphen if applicable
    if lastword[-1] == "-":
      lastword = lastword[:-1]
    else:
      continue # No hyphenation, we can already continue

    if is_correct(lastword) or idx >= len(lines) - 1:
      continue # Correct in either the list or the dictionary

    # Likely wrong word. Take a peek at the next line
    next_idx = idx + 1
    if lines[next_idx].strip() == "" and next_idx + 1 < len(lines):
      next_idx += 1 # The the next-next line. Sometimes this happens, especially in the older records

    nextline = lines[next_idx]
    nextword = nextline.split(" ")[0]

    if nextword == "" or nextword[0].isupper():
      continue # Possibly empty line. We also check that the next word is not capitalised

    concat = lastword + nextword
    # Account for "someword,"
    if nextword[-1].lower() not in set("abcdefghijklmnopqrstuvwxyz"):
      concat = lastword + nextword[:-1]

    if not is_correct(concat):
      continue # Neither correct for the word list, nor for the dictionary

    # Now the word is correct, indicating there was hyphenation in the original
    # document. So what we need to do now is remove that first word from the next
    # line and slap it to the last word of this line.
    words[-1] = concat
    lines[idx] = " ".join(words)
    lines[next_idx] = " ".join(nextline.split(" ")[1:])

  # SECOND ITERATION: NOW BACKWARD SCANNING
  for idx, line in enumerate(lines):
    if line == "" or " " not in line:
      continue

    words = line.split(" ")

    if len(words[0]) == 0 or words[0][0].isupper():
      continue # The word is capitalised, indicating that this is not likely a hyphenation.

    if is_correct(words[0]) or idx == 0:
      continue # Correct in either the list or the dictionary

    # Likely wrong word. Take a peek at the previous line
    prev_index = idx - 1
    if lines[prev_index].strip() == "" and prev_index > 0:
      prev_index -= 1 # As in the forward scan, we CAN skip an empty line

    prevline = lines[prev_index]
    lastword = prevline.split(" ")[-1]

    if lastword.strip() == "":
      continue # Possibly empty line.

    # Remove a hyphen if applicable
    if lastword[-1] == "-":
      lastword = lastword[:-1]
    else:
      continue # No hyphenation, we can already continue

    concat = lastword + words[0]
    if not is_correct(concat):
      continue # Neither correct for the word list, nor for the dictionary

    # Now the word is correct, indicating there was hyphenation in the original
    # document. So what we need to do now is remove that last word from the
    # previous line and slap it to the first word of this line.
    words[0] = concat
    lines[idx] = " ".join(words)
    lines[prev_index] = " ".join(prevline.split(" ")[:-1])

  # Now return the corrected document
  return "\n".join(lines)

if __name__ == "__main__":
  files = [
    "fdp_btw2021.txt",
    "gruene_btw2021.txt",
    "spd_btw2021.txt"
  ]

  for file in files:
    print(f"Dehyphenating {file}...")
    with open(file, "r") as fp:
      text = fp.read()
    
    text = dehyphenate(text)

    with open(f"dehyph.{file}", "w") as fp:
      fp.write(text)

  # At the end, make sure to save the vocab
  with open("vocab_final.txt", "w") as fp:
    for word in vocab:
      print(word, file=fp)
