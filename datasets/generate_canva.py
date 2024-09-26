from  PIL import Image, ImageFilter, ImageDraw
import numpy as np
import os 
import cv2
from numpy.random import uniform, choice
from random import randint, choice as rand_choice
current_dir = os.path.dirname(os.path.abspath(__file__))

if "dataset" not in current_dir:
    current_dir = os.path.join(current_dir, "dataset")
else:
    current_dir = os.path.join(current_dir, ".")


BACKGROUND_PATH  =  os.path.join(current_dir, 'resources/background')
BACKGROUND_LABEL = 0
BACKGROUND_COLOR = (50, 50, 50)
BLURED_BORDER_WIDTH_RANGE = (1, 7)
GAUSSIAN_NOISE_STD_RANGE = (2, 10)
NOISE_PATTERN = ['border_hole', 'center_hole', 'corner_hole', 'phantom_character']
NOISE_PATH = os.path.join(current_dir, 'resources/noise_pattern')
NB_NOISE_PATTERN = (0,5)
BACKGROUND_BLURED_BORDER_WIDTH_RANGE = (1, 10)
BLUR_RADIUS_RANGE = (0.2, 0.4)
LINE_OPACITY_RANGE = (100, 255)
LINE_STD_GAUSSIAN_NOISE_RANGE = (4, 40)
LINE_WIDTH_RANGE = (1, 4)

BLACK_AND_WHITE_FREQ = 0.5
COMMON_FONT_FREQ = 0.5
CONTEXT_BACKGROUND_FREQ = 0.3
DOUBLE_PAGE_FREQ = 0.3
DOUBLE_COLUMN_FREQ = 0.3
NOISE_PATTERN_SIZE_RANGE = {
    'border_hole': (5, 30),
    'center_hole': (5, 30),
    'corner_hole': (5, 30),
    'phantom_character': (5, 50),
}
NOISE_PATTERN_OPACITY_RANGE = (0.2, 0.6)
POS_ELEMENT_OPACITY_RANGE = {
    'drawing': (200, 255),
    'glyph': (150, 255),
    'image': (150, 255),
    'table': (200, 255),
    'text': (200, 255),
}
NEG_ELEMENT_OPACITY_RANGE = {
    'drawing': (0, 10),
    'glyph': (0, 10),
    'image': (0, 25),
    'table': (0, 10),
    'text': (0, 10),
}
NEG_ELEMENT_BLUR_RADIUS_RANGE = (1, 2.5)

BACKGROUND_BLUR_RADIUS_RANGE = (0, 0.2)
BACKGROUND_COLOR_BLEND_FREQ = 0.1
CONTEXT_BACKGROUND_UNIFORM_FREQ = 0.5
DRAWING_CONTRAST_FACTOR_RANGE = (1, 4)
DRAWING_WITH_BACKGROUND_FREQ = 0.3
DRAWING_WITH_COLOR_FREQ = 0.3
GLYPH_COLORED_FREQ = 0.5
LINE_WIDTH_RANGE = (1, 4)
TABLE_LAYOUT_RANGE = {
    'col_size_range': (50, 200),
}

TEXT_BASELINE_HEIGHT = 5
TEXT_BBOX_FREQ = 0.3
TEXT_BBOX_BORDER_WIDTH_RANGE = (1, 6)
TEXT_BBOX_PADDING_RANGE = (0, 20)
TEXT_COLORED_FREQ = 0.5
TEXT_FONT_TYPE_RATIO = {
    'arabic': 0,
    'chinese': 0,
    'handwritten': 0.5,
    'normal': 0.5,
}
TEXT_JUSTIFIED_PARAGRAPH_FREQ = 0.7
TEXT_ROTATION_ANGLE_RANGE = (-60, 60)
TEXT_TIGHT_PARAGRAPH_FREQ = 0.5
TEXT_TITLE_UPPERCASE_RATIO = 0.5
TEXT_TITLE_UNILINE_RATIO = 0.25
TEXT_UNDERLINED_FREQ = 0.1
TEXT_UNDERLINED_PADDING_RANGE = (0, 4)

def random_folder(folder):
    paths = os.listdir(folder)
    if '.DS_Store' in paths:
        paths.remove('.DS_Store')
    return paths

def generate_background(width,height):
    size = (width, height)
    label = BACKGROUND_LABEL
    color = BACKGROUND_COLOR
    folder_background = choice(random_folder(BACKGROUND_PATH))
    img_path = choice([os.path.join(BACKGROUND_PATH, folder_background, file) for file in os.listdir(os.path.join(BACKGROUND_PATH, folder_background)) if file.endswith('.jpg')])
    
    img = Image.open(img_path)#.resize(size, resample=Image.ANTIALIAS).convert('RGB')
    ## crop image accoding to size
    
    ## check if img.width < width
    if img.width > width:
        center = (img.width / 2, img.height / 2)
        img = img.crop((center[0] - width / 2, center[1] - height / 2, center[0] + width / 2, center[1] + height / 2))
    else:
        ## first resize img to 
        new_width = width
        new_height = int(new_width / width * height)
        img = img.resize((new_width,new_height))
        center = (img.width / 2, img.height / 2)
        img = img.crop((center[0] - width / 2, center[1] - height / 2, center[0] + width / 2, center[1] + height / 2))
        
    img = img.resize(size, resample=Image.ANTIALIAS).convert('RGB')
    blur_radius = uniform(*BACKGROUND_BLUR_RADIUS_RANGE)

    pos_x, pos_y = (0, 0)
    color_blend = choice([True, False], p=[BACKGROUND_COLOR_BLEND_FREQ, 1 - BACKGROUND_COLOR_BLEND_FREQ])
    flip = choice([True,False])
    if color_blend:
        new_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
        new_img[:, :, 0] = randint(0, 360)
        try:
            img = Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB))
        except Exception as e:
            print('new_img',new_img.shape,img_path)
            print("An error occurred:", e)
    if flip:
        img =  img.transpose(Image.FLIP_LEFT_RIGHT).filter(ImageFilter.GaussianBlur(blur_radius))
    else:
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))

    border_width = randint(*BACKGROUND_BLURED_BORDER_WIDTH_RANGE)
    return {'img': img, 'position': (pos_x, pos_y), 'size': size, 'border_width': border_width}


def resize(img, size, keep_aspect_ratio=True, resample=Image.ANTIALIAS):
    return img.resize(size, resample=resample)
    
def get_random_noise_pattern(width, height):
    pattern_type = choice(NOISE_PATTERN)
    folder_noise = os.path.join(NOISE_PATH, pattern_type) 
    
    pattern_path = choice([os.path.join(NOISE_PATH, folder_noise, file) for file in os.listdir(os.path.join(NOISE_PATH, folder_noise)) if file.endswith('.png')])


    img = Image.open(pattern_path).convert('L')

    ## change it 
    size_min, size_max = NOISE_PATTERN_SIZE_RANGE[pattern_type]

    size_max = min(min(width, height), size_max)
   
    if size_max == 0:
        print(width, height)
    size = (randint(size_min, size_max), randint(size_min, size_max))
 
    if pattern_type in ['border_hole', 'corner_hole']:
        try:
            img =  resize(img, size, keep_aspect_ratio=True, resample=Image.ANTIALIAS)
        except Exception as e:
            print('size',img.size,size, size_min)
            print("An error occurred:", e)
        rotation = choice([None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
        if rotation is not None:
            img = img.transpose(rotation)
        if pattern_type == 'border_hole':
            if rotation is None:
                position = ((randint(0, width - img.size[0]), 0))
            elif rotation == Image.ROTATE_90:
                position = (0, randint(0, height - img.size[1]))
            elif rotation == Image.ROTATE_180:
                position = ((randint(0, width - img.size[0]), height - img.size[1]))
            else:
                position = (width - img.size[0], randint(0, height - img.size[1]))
        else:
            if rotation is None:
                position = (0, 0)
            elif rotation == Image.ROTATE_90:
                position = (0, height - img.size[1])
            elif rotation == Image.ROTATE_180:
                position = (width - img.size[0], height - img.size[1])
            else:
                position = (width - img.size[0], 0)
    else:
        img = resize(img, size, keep_aspect_ratio=False, resample=Image.ANTIALIAS)
        rotation = randint(0, 360)
        img = img.rotate(rotation, fillcolor=255)
        pad = max(img.width, img.height)
        position = (randint(0, max(0, width - pad)), randint(0, max(0, height - pad)))

    alpha = uniform(*NOISE_PATTERN_OPACITY_RANGE)
    arr = np.array(img.convert('RGBA'))
    arr[:, :, 3] = (255 - arr[:, :, 2]) * alpha
    hue_color = randint(0, 360)
    value_ratio = uniform(0.95, 1)
    return Image.fromarray(arr), hue_color, value_ratio, position

def generate_random_noise_patterns(background):
    patterns, positions = [], []
    bg_width, bg_height = background['size']
    bg_x, bg_y = background['position']
    for _ in range(randint(*NB_NOISE_PATTERN)):
        pattern, hue_color, value_ratio, position = get_random_noise_pattern(bg_width, bg_height)
        position = (position[0] + bg_x, position[1] + bg_y)
        patterns.append((pattern, hue_color, value_ratio))
        positions.append(position)
    return patterns, positions

def draw_noise_patterns( canvas,noise_patterns):
    for (noise, hue_color, value_ratio), pos in zip(*noise_patterns):
        patch = np.array(noise)
        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        patch_hsv[:, :, 0] = hue_color
        patch_hsv[:, :, 2] = patch_hsv[:, :, 2] * value_ratio
        new_patch = Image.fromarray(cv2.cvtColor(patch_hsv, cv2.COLOR_HSV2RGB))
        canvas.paste(new_patch, pos, mask=noise)

def paste_with_blured_borders(canvas, img, position=(0, 0), border_width=3):
    canvas.paste(img, position)
    mask = Image.new('L', canvas.size, 0)
    blur = canvas.filter(ImageFilter.GaussianBlur(border_width / 2))
    canvas.paste(blur, mask=mask)



def to_image(text_img,background,noise_patterns):
    size = text_img['img'].size
    canvas = Image.new(mode='RGB', size = size)
    background_img = background['img']
    paste_with_blured_borders(canvas, background_img, background['position'], background['border_width'])
    draw_noise_patterns(canvas,noise_patterns)

    canvas.paste(text_img['img'], text_img['position'], text_img['img'])


    blur_radius = uniform(*BLUR_RADIUS_RANGE)
    black_and_white = choice([True, False], p=[BLACK_AND_WHITE_FREQ, 1 - BLACK_AND_WHITE_FREQ])
    if blur_radius > 0:
         canvas = canvas.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    if black_and_white:
         canvas = canvas.convert("L").convert("RGB")

    return canvas

def generate_canva(text_img):
    size = text_img['img'].size
    background = generate_background(size[0],size[1])
    noise_patterns = generate_random_noise_patterns(background)
    new_img = to_image(text_img,background,noise_patterns)
    return new_img
