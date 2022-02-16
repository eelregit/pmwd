"""Draw pmwd in the Nova font."""

from PIL import Image, ImageFont, ImageDraw


font = '../docs/nova/NovaRoundSlim-BookOblique.ttf'
font = ImageFont.truetype(font=font, size=128)

text = 'pmwd'
print('text bbox:', font.getbbox(text), font.getbbox(text, anchor='mm'))

size = (392, 128)
xy = (189, 51)

alpha = Image.new('L', size)
draw = ImageDraw.Draw(alpha)
draw.text(xy, text, fill='white', font=font, anchor='mm')

image = Image.new('L', size, color='black')
image.putalpha(alpha)

image.save('nova.png')
