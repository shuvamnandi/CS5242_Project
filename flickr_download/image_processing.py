from PIL import Image

from torchvision import transforms

resized_image_height = 28
resized_image_width = 28


def get_resize_image_tensor(file_name):
    img = Image.open(file_name)
    tensor_converter = transforms.ToTensor()
    resize_img = img.resize((resized_image_height, resized_image_width))
    resize_img_tensor = tensor_converter(resize_img)
    return resize_img_tensor


def main():
    image_tensor = get_resize_image_tensor('../data/CMFD/00000/00000_Mask.jpg')
    print(image_tensor.size())


if __name__ == '__main__':
    main()