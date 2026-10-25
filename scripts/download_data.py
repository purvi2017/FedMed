from pathlib import Path

from monai.apps import download_and_extract

URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
MD5 = "240a19d752f0d9e9101544901065d872"


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    target = data_dir / "Task01_BrainTumour"
    if target.exists():
        print(f"Dataset already present at {target}")
        return

    archive = data_dir / "Task01_BrainTumour.tar"
    download_and_extract(URL, str(archive), str(data_dir), hash_val=MD5, hash_type="md5")
    print(f"Dataset ready at {target}")


if __name__ == "__main__":
    main()