import requests
import tqdm
import os


def freudenberg2022(filename):
    """
    Downloads model weights for a pre-trained model on 20cm imagery from a URL with a progress bar.

    Original paper:
    Freudenberg, M., Magdon, P. and Nölke, N., 2022.
    Individual tree crown delineation in high-resolution remote sensing images based on U-Net.
    Neural Computing and Applications, 34(24), pp.22197-22207.

    Args:
        filename (str): The name of the file to save the downloaded content to.
    """
    url = "https://syncandshare.desy.de/index.php/s/NcFgPM4gX2dtSQq/download/lUnet-resnet18_epochs=209_lr=0.0001_width=224_bs=32_divby=255_custom_color_augs_k=3_jitted.pt"
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1KB

    print("Downloading label shape files tile # 32_704_5708_2...")
    print(
        "Freudenberg, M., Magdon, P. and Nölke, N., 2022. \nIndividual tree crown delineation in high-resolution remote sensing images based on U-Net. \nNeural Computing and Applications, 34(24), pp.22197-22207. \nhttps://doi.org/10.1007/s00521-022-07640-4"
    )

    progress_bar = tqdm.tqdm(total=total_size, unit="iB", unit_scale=True)

    with open(filename, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR, something went wrong")
