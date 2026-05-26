import argparse
from training.train import train


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str)

    args = parser.parse_args()

    if args.stage == "train":
        train()


if __name__ == "__main__":
    main()