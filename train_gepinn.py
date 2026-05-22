import sys

from train_gabor_pinn import main


if __name__ == "__main__":
    if "--family" not in sys.argv:
        sys.argv.extend(["--family", "gepinn"])
    main()
