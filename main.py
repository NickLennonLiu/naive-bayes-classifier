import argparse
import yaml

from dataloader import Dataloader
from params import get_args


def main():
    args = get_args()
    dataloader = Dataloader(args)


if __name__ == "__main__":
    main()
