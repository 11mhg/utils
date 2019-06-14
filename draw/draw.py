import numpy as np
import os, random
import cv2
import colorsys
from PIL import Image, ImageDraw, ImageFont
from pipeline.bbox import Box


def get_colors_for_classes(num_classes):
    if (hasattr(get_colors_for_classes, "colors") and
        len(get_colors_for_classes.colors) == num_classes):
        return get_colors_for_classes.colors

    hsv_tuples = [(x/num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] *255)),
                colors))
    random.seed(10101)
    random.shuffle(colors)
    random.seed(None)
    get_colors_for_classes.colors = colors
    return colors



def draw(image, boxes, classes):
    if image.max() <= 1.0:
        image = Image.fromarray(np.floor(image*255.+0.5).astype('uint8'))
    else:
        image = Image.fromarray(image.astype('uint8'))
    font_dir = os.path.expanduser('~/utils/draw/font/FiraMono-Medium.otf')
    font = ImageFont.truetype(
            font = font_dir,
            size=np.floor(3e-2*image.size[1]+0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1])//300

    colors = get_colors_for_classes(len(classes))

    for box in boxes:
        box_class = classes[box.label]
        if box.score is not None:
            score = box.score
            label = '{} {:.2f}'.format(box_class,score)
        else:
            label = '{}'.format(box_class)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label,font)

        top, left, bottom, right = box.y0, box.x0, box.y1, box.x1
        top = max(0,np.floor(top+0.5).astype('int32'))
        left = max(0,np.floor(left+0.5).astype('int32'))
        bottom = min(image.size[1],np.floor(bottom+0.5).astype('int32'))
        right = min(image.size[0],np.floor(right+0.5).astype('int32'))
#        print(label, (left, top), (right, bottom))
        if top - label_size[1] >=0:
            text_origin = np.array([left,top-label_size[1]])
        else:
            text_origin = np.array([left, top+1])

        for i in range(thickness):
            draw.rectangle(
                    [left+i, top+i, right-i, bottom-i],
                    outline = colors[box.label])
        draw.rectangle(
                [tuple(text_origin),tuple(text_origin+label_size)],
                fill=colors[box.label])
        draw.text(text_origin, label, fill=(0,0,0),font=font)
        del draw
    image = np.array(image,dtype=np.float32)
    image = image/255. if image.max() >= 1.0 else image

    return image
