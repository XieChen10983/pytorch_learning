# coding=gbk
from torchvision import transforms
from train_code import run_style_transfer
from load_img import load_img, show_img

style_img = load_img('./picture/style.png')
content_img = load_img('./picture/content.png')
# show_img(style_img)
input_img = content_img.clone()
out = run_style_transfer(content_img, style_img, input_img, num_epochs=200)
save_pic = transforms.ToPILImage()(out.squeeze(0))
save_pic.save('./picture/saved_picture.png')
