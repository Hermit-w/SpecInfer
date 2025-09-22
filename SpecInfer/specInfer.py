import argparse

from SpecInfer.model_loader.loader import ModelLoader


def main(args_: argparse.Namespace):
    model_path = args_.model
    print(model_path)
    pass


if __name__ == '__main__':
    args = argparse.ArgumentParser("SpecInfer")
    args.add_argument("--model", "-m", type=str, required=True, help="The path/name of the model.")
    args.add_argument("--draft_model", "-dm", type=str, help="The draft model for speculative decode.")
    args.add_argument("--tensor_parallel", "-tp", type=int, default=1, help="Tensor parallel size.")
    args.add_argument("--tensor_parallel_draft", "-dtp", type=int, default=1, help="Tensor parallel size of draft model.")

    args_ = args.parse_args()
    main(args_)
