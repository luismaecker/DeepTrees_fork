import requests
import tqdm
import os


def load_tiles(zip_filename):
    """
    Downloads subtiles of DOP20 tile # 32_704_5708_2 from a URL with a progress bar.

    Args:
        url: The URL of the file to download.
        filename: The name of the file to save the downloaded content to.
    """
    url = "https://syncandshare.desy.de/index.php/s/9MntrHN3xJoQXiH/download/tiles.zip"
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1KB
    print("Downloading subtiles for tile # 32_704_5708_2...")
    print(
        "Halle DOP20: © GeoBasis-DE / LVermGeo ST 2022. https://www.lvermgeo.sachsen-anhalt.de/de/gdp-digitale-orthophotos.html"
    )
    progress_bar = tqdm.tqdm(total=total_size, unit="iB", unit_scale=True)

    with open(zip_filename, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR, something went wrong")


def load_labels(zip_filename):
    """
    Downloads all the labels ESRI shape files from a URL with a progress bar.

    Args:
        url: The URL of the file to download.
        filename: The name of the file to save the downloaded content to.
    """
    url = "https://syncandshare.desy.de/index.php/s/C3WcwXYZLNSJxbE/download/labels.zip"
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1KB

    print("Downloading label shape files tile # 32_704_5708_2...")
    print("Taimur Khan 2024. © Helmholtz-UFZ.")

    progress_bar = tqdm.tqdm(total=total_size, unit="iB", unit_scale=True)

    with open(zip_filename, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR, something went wrong")
