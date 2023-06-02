import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--m", type=float, default=20, help="m  argument used in the model.")
    parser.add_argument("-hi", "--hidden", type=int, default=40, help="The number of hidden dimensions of the model.")
    parser.add_argument("-conv", "--conv", type=str2bool, default=False, help="Conv net?")
    parser.add_argument("-num", "--number_verifications", type=int, default=100, help="Number of images we want to verify.")
    parser.add_argument("-p", "--path", type=str, default="models/mon_h40_m20.0.pt", required=False, help="Path to the model w.r.t. directory root.")
    parser.add_argument("-aa", "--adaptive_alpha", action="store_true", help="Use a heuristic to find an alpha which leads to nice convergence in the forward-backward iteration.")
    parser.add_argument("-i", "--iters", type=int, default=200, help="Zonotope iterations for finding an AFP.")
    parser.add_argument("-a", "--alpha", type=float, default=1 / 32, help="Base alpha value.")
    parser.add_argument("-e", "--eps", type=float, default=2 / 255, help="Epsilon L-inf we want to verify.")
    parser.add_argument("-ex", "--explicit", type=int, default=-1, help="Id of an explicit example we want to verify.")
    parser.add_argument("-ps", "--proj_start", type=int, default=3, help="Iteration at which we start the projection to the PCA basis.")
    parser.add_argument("-dil", "--dilation_new_base", type=int, default=10, help="Number of epochs between updates of the projection basis.")
    parser.add_argument("-fpi", "--fixed_point_init", action="store_true", help="Initialize the model at the Fixed-Point z* of the center-image.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output.")
    parser.add_argument("-d", "--double", action="store_true", help="Use double precision")
    parser.add_argument("-ab", "--adaptive_basis", action="store_true", help="When re-computing the basis remove unused vectors and replace them by basis vectors which best represent the worst represtend old vectors.")
    parser.add_argument("-pr", "--peaceman_rachford", action="store_true", help="Use Peaceman-Rachford iterations instead of Forward-Backward.")
    parser.add_argument("-wid", "--widening_iter", type=int, default=0, help="Start with widening iterations before we switch upon convergence. 0 for no widening. negative for const widening")
    # parser.add_argument("-k", "--k_step", type=int, default=1, help="Number of zonotopes that you want to go back for containment checks") # Set to 10
    parser.add_argument("-chk", "--check_sound", type=str2bool, default=False, help="Check the soundness via sampling checks.")
    parser.add_argument("-adv", "--adv_attack", type=str2bool, default=False, help="Check robustness via adv attacks.")
    parser.add_argument("-post_d", "--post_proj_dil", type=int, default=30, help="Dilation of projection after containment was found.")
    parser.add_argument("-pre_d", "--pre_proj_dil", type=int, default=3, help="Dilation of projection before containment was found.")
    parser.add_argument("-jp", "--joint_projection", type=str2bool, default=True, help="Whether to project jointly when using pr or individually.")
    parser.add_argument("-sfwbw", "--switch_to_fwbw", type=str2boolOrFloat, default=None, help="To switch from pr to fwbw after containment, indicate alpha else None. Set to true if you want to use adaptive alpha")
    parser.add_argument("-log", "--log", type=str, default=None, help="File for logging. No logging if None.")
    parser.add_argument("-os", "--optimize_slopes", type=int, default=0, help="Whether to optimize the slopes of the ReLU transformer: 10 for in loop optimization, +1 for first level unrolled, +2 for second level unrolled")
    parser.add_argument("-domain", "--domain", type=str, default="LBZonotope", help="Which domain to use.")

    parser.add_argument("-k", "--kernel_x", type=int, default=6, help="Kernel size for input conv")
    parser.add_argument("-s", "--stride_x", type=int, default=3, help="Stride for input conv")
    parser.add_argument("-ds", "--dataset", type=str, default="mnist", help="Dataset to use")
    
    parser.add_argument("-nb", "--no_box_term", type=str2bool, default=False, help="Do not use box term of LBZonotope.")
    parser.add_argument("-rc", "--require_contained", type=str2bool, default=False, help="Require to show containment multiple times.")

    parser.add_argument("--verify_range", type=str2bool, default=False, help="Spec range to verify.")
    parser.add_argument('--range_x', nargs='+', type=float, default=[0.0, 0.0], help='x_range to verify')
    parser.add_argument('--range_y', nargs='+', type=float, default=[0.0, 0.0], help='x_range to verify')
    parser.add_argument('--range_psi', nargs='+', type=float, default=[0.0, 0.0], help='x_range to verify')
    parser.add_argument('--range_depth', type=int, default=3, help='Max recursion depth')

    parser.add_argument("--unsound_zono", type=str2bool, default=False, help="Whether to run unsound mirror zono." )
    parser.add_argument("--containment_experiment", type=str2bool, default=False, help="Whether to run containment experiments.")

    parser.add_argument(
        '--lben',
        action='store_true',
        help='Whether to have LBEN parameterization.')
    parser.add_argument(
        '--lben_cond',
        type=float,
        default=3,
        help='Condition number of diagonal matrix in LBEN parameterization.')
    args = parser.parse_args()

    if args.log is None or args.log in ["none", "None"]:
        args.log = None

    if args.adaptive_alpha:
        pass
        if not args.switch_to_fwbw:
            raise argparse.ArgumentTypeError('To use adaptive alpha, switch_to_fwbw must be set.')
    else:
        if type(args.switch_to_fwbw) is bool:
            raise argparse.ArgumentTypeError('Used boolean switch_to_fwbw but no adaptive-alpha.')

    if args.verify_range:
        assert args.dataset.startswith("HCAS")
        assert len(args.range_x) == len(args.range_y) == len(args.range_psi) == 2

    return args

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2boolOrFloat(v):
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        try:
            return float(v) 
        except:
            raise argparse.ArgumentTypeError('Boolean value or float expected.')

def str2boolOrInt(v):
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        try:
            return int(v)
        except:
            raise argparse.ArgumentTypeError('Boolean value or float expected.')