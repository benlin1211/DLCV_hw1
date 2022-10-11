import gdown

url = "https://drive.google.com/file/d/1uJsD43PF_YU-8LIE3yPd51NDj5zmZ01m/view?usp=sharing"
output = "ckpt.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)



url = "https://drive.google.com/file/d/1s8irubk-MYhERpJOb_jYJ2SFwKElvqKg/view?usp=sharing"
output = "ckpt1-2.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)


url = "https://drive.google.com/file/d/1f4aMq6jCXodYuttLNOLk92vOymBRzb6m/view?usp=sharing"
output = "ckpt_1-2-ensemble.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

