################################################################################
# Copyright 2024-2025 Lawrence Livermore National Security, LLC and other
# Heesung Shim (shim2@llnl.gov)
# See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Main app to run training or testing based on the run mode.
# This code also loads the YAML configuration file to set parameters.
################################################################################
import os
import argparse
import yaml

from attention_trainer_val import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-mode", type=int, help="1: train, 2: test, 3: extract latent features, 11: train MIL, 12: test MIL")
    parser.add_argument("--config", help="configuration file")
    return parser.parse_args()

def load_config(config_fn):
    with open(config_fn, "r") as stream:
        try:
            yenv = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return yenv

def main():
    args = parse_args()
    configs = load_config(args.config)

    trainer = Trainer(configs, args.run_mode)
    if args.run_mode == 1:
        trainer.train()
    elif args.run_mode == 2:
        trainer.test()
    else:
        print("Invalid run mode")

if __name__ == "__main__":
    main()