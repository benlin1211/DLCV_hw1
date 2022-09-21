import os
import argparse
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="hw 1-1 inference",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("src", help="Source data location")
    parser.add_argument("dest", help="Destination location")
    args = parser.parse_args()
    # config = vars(args)


    print("hello")
    print(args.src)
    print(args.dest)