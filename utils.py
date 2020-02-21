import sys
import imageio


def process_command_args(arguments):

    phone = ""
    resolution = "orig"
    use_gpu = "true"
    model = "sony"

    for args in arguments:

        if args.startswith("phone"):
            phone = args.split("=")[1]

        if args.startswith("resolution"):
            resolution = args.split("=")[1]

        if args.startswith("use_gpu"):
            use_gpu = args.split("=")[1]

        if args.startswith("model"):
            model = args.split("=")[1]

    if phone == "":
        print("\nPlease specify the camera model by running the script with the following parameter:\n")
        print("python run_model.py phone={iphone,blackberry,sony,custom}\n")
        sys.exit()

    if phone not in ["iphone", "sony", "blackberry", "custom"]:
        print("\nPlease specify the correct camera model:\n")
        print("python run_model.py phone={iphone,blackberry,sony,custom}\n")
        sys.exit()

    if resolution not in ["orig", "high", "medium", "small", "tiny"]:
        print("\nPlease specify the correct resolution:\n")
        print(
            "python run_model.py phone={iphone,blackberry,sony,custom} resolution={orig,high,medium,small,tiny}\n")
        sys.exit()

    if use_gpu not in["true", "false"]:
        print("\nPlease specify correctly the gpu usage:\n")
        print(
            "python run_model.py phone={iphone,blackberry,sony,custom} use_gpu={true,false}\n")
        sys.exit()
    
    if model not in ["iphone", "sony", "blackberry", "pynet_level_0.ckpt"]:
        print("\nPlease specify the correct pre-trained model:\n")
        print("python run_model.py model={iphone,blackberry,sony,pynet_level_0.ckpt}\n")
        sys.exit()

    return phone, resolution, use_gpu, model


def get_resolutions(image):

    # IMAGE_HEIGHT, IMAGE_WIDTH

    res_sizes = {}

    res_sizes["iphone"] = [1536, 2048]
    res_sizes["blackberry"] = [1560, 2080]
    res_sizes["sony"] = [1944, 2592]
    res_sizes["high"] = [1260, 1680]
    res_sizes["medium"] = [1024, 1366]
    res_sizes["small"] = [768, 1024]
    res_sizes["tiny"] = [600, 800]
    res_sizes["custom"] = list(imageio.imread(image, pilmode="RGB").shape)[:-1]

    return res_sizes


def get_specified_res(res_sizes, phone, resolution):

    if resolution == "orig":
        IMAGE_HEIGHT = res_sizes[phone][0]
        IMAGE_WIDTH = res_sizes[phone][1]
    else:
        IMAGE_HEIGHT = res_sizes[resolution][0]
        IMAGE_WIDTH = res_sizes[resolution][1]

    IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3

    return IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE


def extract_crop(image, resolution, phone, res_sizes):

    if resolution == "orig":
        return image

    else:

        x_up = int((res_sizes[phone][1] - res_sizes[resolution][1]) / 2)
        y_up = int((res_sizes[phone][0] - res_sizes[resolution][0]) / 2)

        x_down = x_up + res_sizes[resolution][1]
        y_down = y_up + res_sizes[resolution][0]

        return image[y_up: y_down, x_up: x_down, :]
