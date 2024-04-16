import argparse


def arg_parser():

    parser = argparse.ArgumentParser()

    # =========== Optimizer settings ======================
    parser.add_argument('--optimizer', type=str, default="Adam",
                        choices=['Adam', 'SGD'],
                        help='Optimizer for training')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum of optimizer.')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Initial learning rate.')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay.')
    
    parser.add_argument('--epochs', type=int,  default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--eval_freq', type=int,  default=1,
                        help='Number of epochs to train.')
    parser.add_argument('--onlyUnlabel', type=str,
                        choices=['yes', 'no'],
                        default='yes')
    
    # =========== CR settings ======================
    parser.add_argument('--cr_loss', type=str,
                        choices=['kl', 'l2'],
                        default='kl')
    parser.add_argument('--cr_tem', type=float, default=1,
                        help='Temperature for CR.')
    parser.add_argument('--cr_conf', type=float, default=0.5,
                        help='Confidence level for CR.')
    
    parser.add_argument('--lambda_cr', type=float, default=0.5,
                        help='lambda cr')
    
    parser.add_argument('--lambda_ce', type=float, default=0.5,
                        help='lambda ce.')
    
    parser.add_argument('--lambda_g', type=float, default=1,
                        help='lambda g.')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='beta for prediction.')

    # =========== GPR settings ======================
    parser.add_argument('--lg_k', type=int, default=10,
                        help='k hop for one view')
    parser.add_argument('--gg_k', type=int, default=20,
                        help='k hop for another view')
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--Init', type=str,
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='PPR')
    parser.add_argument('--alpha',type=float, default=0.1)

    parser.add_argument('--ppnp', default='GPR_prop',
                        choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--Gamma', default=None)

    # =========== Neural network settings ======================
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')

    # =========== Dataset settings ======================
    parser.add_argument('--dataset', type=str, default="MM COVID",
                        choices=['MM COVID', 'ReCOVery','MC Fake', 'LIAR', 'PAN2020'], help='dataset')
    parser.add_argument("--tr", type=float, default=0.8,
                        help='rate of training data')
    parser.add_argument("--vr", type=float, default=0.1,
                        help='rate of validation data')
    parser.add_argument("--num_topics", type=int,
                        help='number of topic nodes')

    
    # =========== Other general settings ======================
    parser.add_argument('--verbose', choices=["True", "False"], default="False",
                        help='show tqdm bar?')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--patience', type=int, default=80)

    parser.add_argument('--fold', type=int, default=10)
    parser.add_argument('--seed', type=int, default=123)


    args = parser.parse_args()
    
    return args