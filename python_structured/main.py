
from load_data import load_data_with_folds, load_param_grid
from experiment import run_param_search
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type="str", required=True, help="JSON file with param grid")
    parser.add_argument("--output_dir", type="str", default="runs", help="Where to safe results")
    return parser.parse_args()

n_folds = 10
random_state = 42

def main():
    args = parse_args()

    X, y, patient_ids, fold_indices = load_data_with_folds(
        file_path="data/slice_localization_data.csv",
        target_col="reference",
        drop_cols=["patientId"],
        n_folds=n_folds,
        random_state=random_state
    )

    param_grid = load_param_grid(args.config)

    best_model, best_params, best_score, results = run_param_search(
        X, y, patient_ids, fold_indices, param_grid, n_folds
    )

    print("Best L2 Error:", best_score)
    print("Bet params:", best_params)

    save_run(args.output_dir, best_model, best_params, best_score, results)

if __name__ == "__main__":
    main()