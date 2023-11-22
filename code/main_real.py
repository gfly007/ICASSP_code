from datasets import get_dataset
from train_causal import train_causal_real
import opts
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    args = opts.parse_args()
    dataset_name, feat_str, _ = opts.create_n_filter_triples([args.dataset])[0]
    dataset = get_dataset(dataset_name, sparse=True, feat_str=feat_str, root=args.data_root)  # 数据生成

    model_func = opts.get_model(args)  # 模型设计

    train_causal_real(dataset, model_func, args)  # 训练
