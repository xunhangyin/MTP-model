import argparse
def parser_add_main_args(parser):
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval_step', type=int,
                        default=1000, help='how often to print')
    parser.add_argument('--model_path', type=str, default="./ESM2150M/")
    # hyper_parameter for model arch and training
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--accumulate_steps', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--type', type=str,default="BPO")
    parser.add_argument("--output_dir",type=str,default="./model_output2/")