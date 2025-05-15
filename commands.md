docker compose run --rm base bash -c "cd /app/dig-kongzii/examples/ggraph/GraphAF && CUDA_VISIBLE_DEVICES=0 python run_rand_gen.py --train_path /app/data/molecules/QM99/train_smiles.txt --num_max_node 9 --save_interval 10 --n_to_gen 1000"

docker compose run --rm base bash -c "cd /app/dig-kongzii/examples/ggraph/GraphDF && CUDA_VISIBLE_DEVICES=1 python run_rand_gen.py --train_path /app/data/molecules/QM99/train_smiles.txt --num_max_node 9 --save_interval 10 --n_to_gen 1000"
