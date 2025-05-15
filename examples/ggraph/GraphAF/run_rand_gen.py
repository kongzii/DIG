import json
import dagshub
import os
import argparse
import random
from rdkit import RDLogger
from torch_geometric.loader import DenseDataLoader
from dig.ggraph.dataset import QM9, ZINC250k, MOSES, SmilesDataset
from dig.ggraph.method import GraphAF
from dig.ggraph.evaluation import RandGenEvaluator
from nmln import utils

RDLogger.DisableLog('rdApp.*')

dagshub.init(
    repo_owner=os.environ.get("DAGSHUB_REPO_OWNER"),
    repo_name=os.environ.get("DAGSHUB_REPO_NAME"),
    mlflow=True,
)

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str)
parser.add_argument('--val_path', type=str, default="")
parser.add_argument('--model_path', type=str, default='./saved_ckpts/rand_gen/rand_gen_qm9.pth', help='The path to the saved model file')
parser.add_argument('--train', action='store_true', default=True, help='specify it to be true if you are running training')
parser.add_argument('--num_max_node', type=int, default=8, help='The size of molecules to be used')
parser.add_argument('--n_to_gen', type=int, default=100, help='The number of molecules to be generated per save_interval')
parser.add_argument('--save_interval', type=int, default=100, help='Every N steps')

args = parser.parse_args()

if not args.val_path:
    args.val_path = args.train_path.replace("train_", "val_")

if 'qm9' in args.train_path.lower():
    with open('config/rand_gen_qm9_config_dict.json') as f:
        conf = json.load(f)
elif 'zinc250k' in args.train_path.lower():
    with open('config/rand_gen_zinc250k_config_dict.json') as f:
        conf = json.load(f)
else:
    print("Only qm9 and zinc250k datasets are supported!")
    exit()

train_dataset = SmilesDataset(args.train_path, args.num_max_node, conf["atom_list"])
val_dataset = SmilesDataset(args.val_path, args.num_max_node, conf["atom_list"])
val_smiles = val_dataset.smiles

runner = GraphAF()

conf["max_epochs"] = 10_000
conf["n_to_gen"] = args.n_to_gen
conf["save_interval"] = args.save_interval
conf["model"]["max_size"] = args.num_max_node
conf["model"]["edge_unroll"] = min(args.num_max_node, 12)
conf["model"]["num_min_node"] = args.num_max_node
conf["model"]["num_max_node"] = args.num_max_node
conf["model"]["temperature"] = conf["temperature"]
conf["model"]["atomic_num_list"] = conf["atom_list"]
conf["num_min_node"] = args.num_max_node
conf["num_max_node"] = args.num_max_node

data_name = args.train_path.split("/")[-2]
run_name = f"{runner.__class__.__name__}-{data_name}-size={args.num_max_node}-n_to_gen={conf['n_to_gen']}-save_int={conf['save_interval']}"

conf["save_dir"] = "output/" + run_name + f"-{random.randint(10000, 99999)}"

logger_mlflow = utils.SafeMLFlowLogger(run_name=run_name)
logger_mlflow.log_metrics({
    f"len/train_dataset": len(train_dataset),
    f"len/val_dataset": len(val_smiles),
})
logger_mlflow.log_hyperparams({
    "data_name": data_name,
    "data_type": "molecules",
    "max_num_atoms": args.num_max_node,
})

if args.train:
    loader = DenseDataLoader(train_dataset, batch_size=conf['batch_size'], shuffle=True)
    runner.train_rand_gen(
        logger_mlflow,
        loader,
        conf["n_to_gen"],
        conf['lr'],
        conf['weight_decay'],
        conf['max_epochs'],
        conf['model'],
        conf['save_interval'],
        conf['save_dir'],
        val_smiles,
        data_name,
    )
else:
    mols, pure_valids = runner.run_rand_gen(conf['model'], args.model_path, args.num_mols, conf['num_min_node'], conf['num_max_node'], conf['temperature'], conf['atom_list'])
    smiles = [data.smile for data in train_dataset]
    evaluator = RandGenEvaluator()
    input_dict = {'mols': mols, 'train_smiles': smiles}

    print('Evaluating...')
    results = evaluator.eval(input_dict)

    print("Valid Ratio without valency check: {:.2f}%".format(sum(pure_valids) / args.num_mols * 100))
