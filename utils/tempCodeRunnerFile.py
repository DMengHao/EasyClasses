
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r"", help='path to dataset')
    parser.add_argument('--proportions', type=float, default=0.8, help='train set proportion')
    parser.add_argument('--shape', type=int, nargs='+', default=[224, 224], help='image shape')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt