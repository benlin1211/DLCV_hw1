import gdown

url = "https://drive.google.com/file/d/1uJsD43PF_YU-8LIE3yPd51NDj5zmZ01m/view?usp=sharing"

output = "ckpt.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)
