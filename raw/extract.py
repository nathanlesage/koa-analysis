# Extracts everything using PDFMiner
from pdfminer.high_level import extract_text
# import pdfminer

if __name__ == '__main__':
  print("Extracting FDP ...")
  fdp = extract_text("fdp_btw2021.pdf")
  with open("fdp_btw2021.txt", "w") as fp:
    fp.write(fdp)
  print("Extracting SPD ...")
  spd = extract_text("spd_btw2021.pdf")
  with open("spd_btw2021.txt", "w") as fp:
    fp.write(spd)
  print("Extracting Grüne ...")
  gruene = extract_text("gruene_btw2021.pdf")
  with open("gruene_btw2021.txt", "w") as fp:
    fp.write(gruene)
  print("Extracting Koalitionsvertrag")
  vertrag = extract_text("koalitionsvertrag_ampel_2021.pdf")
  with open("koalitionsvertrag_ampel_2021.txt", "w") as fp:
    fp.write(vertrag)
  print("Done!")
