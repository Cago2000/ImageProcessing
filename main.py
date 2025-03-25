import image_operations as img_ops

def main():
    img_path = "images/test.jpeg"  # Change this to your image path
    img = img_ops.load_image(img_path)

    if img is not None:
        img_ops.save_image(img, "images/saved_image.jpeg")
        img_ops.delete_image(img_path)
        img_ops.show_image(img)
    else:
        img = img_ops.load_image("images/saved_image.jpeg")
        img_ops.save_image(img, "images/test.jpeg")

if __name__ == "__main__":
    main()