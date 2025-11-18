import logging

from datasets import download_all_datasets


def main():
    logging.basicConfig(format="%(message)s")
    download_all_datasets()


if __name__ == "__main__":
    main()
