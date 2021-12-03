# Create the training data file

from util import parse_preprocessed
from random import shuffle

gruene = [x for x in parse_preprocessed("stanza.dehyph.gruene_btw2021.txt")]
fdp = [x for x in parse_preprocessed("stanza.dehyph.fdp_btw2021.txt")]
spd = [x for x in parse_preprocessed("stanza.dehyph.spd_btw2021.txt")]

maximum = min([len(gruene), len(fdp), len(spd)])

# Now we know the maximum amount of sentences we can add
shuffle(gruene)
shuffle(spd)
shuffle(fdp)

# Cap the size
gruene = gruene[:maximum]
fdp = fdp[:maximum]
spd = spd[:maximum]

all_data = list()
all_data.extend(gruene)
all_data.extend(fdp)
all_data.extend(spd)

# Write to file
print(all_data[0:10])
with open("train_data.txt", "w") as fp:
  for sentences in all_data:
    for rec in sentences:
      fp.write(f"{rec['id']}\t{rec['text']}\t{rec['head']}\t{rec['upos']}\t{rec['deprel']}\t{rec['label']}\n")
