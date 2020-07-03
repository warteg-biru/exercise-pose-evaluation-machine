import os
import sys
from PIL import Image

def join_path(path1, path2):
    return f'{path1}{path2}' if path1[-1:] == '/' else f'{path1}/{path2}'

def get_extension(filename):
    return filename.split('.')[-1]

class ImageCompressor:
    '''
    @param {string} image_path
    '''
    def compress_image(file_path, output_dir):
        picture = Image.open(file_path)
        dimension = picture.size
        print(dimension)
        file_name = file_path.split('/')[-1]
        ext = get_extension(file_name)
        output_path = join_path(output_dir, file_name)
        picture = picture.quantize(method=2)
        picture = picture.resize((720, 405), Image.ANTIALIAS)
        picture.save(output_path, ext.upper())
    

def main():
    input_dir = None
    output_dir = None

    try:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
    except Exception as e:
        print(e)

    if not os.path.exists(output_dir):
        print('Output directory does not exist')
        exit(1)

    for file in os.listdir(input_dir):
        file_path = join_path(input_dir, file)
        img_compressor = ImageCompressor()
        img_compressor.compress_image(file_path, output_dir)

    

if __name__ == '__main__':
    main()
