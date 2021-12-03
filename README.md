# README

This repository contains the code for the analysis of the German Koalitionsvertrag between Social Democrats (SPD), Liberals (FDP) and the Greens (Bündnis 90/Die Grünen) following the 2021 federal election.

## What's in here

In here is everything I used for the analysis sans a few large data files which you will have to download yourself. The first is a set of 300-dimensional German word embeddings. I used the ones from fasttext, but you can use whatever you want. The only requirement is that the file has the following format: One word per line, 301 columns, separated by tab characters, first column contains the word, the remaining 300 the dimensions. Adapt the filename in `classifier.py` if necessary.

Additionally, you'll need some dictionary file with correct German words; see the file `dehyphenate.py` for more info.

## How To Use

I have added the raw data as well as the final output file; please see the different files to see what they do. Most of them are one-shot Python scripts that perform a single task, dependant on other scripts. They need to be run in the following order if you want to reproduce the results which are present in the final data table:

1. Make sure to extract the text from the Wahlprogramme prior to doing anything. Adapt the file paths in `dehyphenate.py` if necessary.
2. Run `dehyphenate.py` (yields the `dehyph` files)
3. Run `stanza.py` (yields the `stanza` files)
4. Run `training_data.py` (yields the `train_data.txt` file)
5. Run `train.py` (trains the model and yields a `model_state.torch` file)
6. Run `predict_koalitionsprogramm.py` (runs the trained classifier over the extracted text by the Koalitionsvertrag and creates the final data table, `koalitionsvertrag_final.tsv`)

The Jupyter Notebook finally contains some plays with the data that I made afterwards to extract the information as required.

## License

The code in this repository is licensed via **GNU GPL v3**. The texts which were used in the analysis are excluded from license since they belong to the respective parties and not to me. The final data table and the analysis notebook (`analysis.ipynb`) are subject to the Creative Commons license **CC-BY-SA 4.0**.
