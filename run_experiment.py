import yaml
from verify_mondeq import verify
import argparse
from bunch import Bunch

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to read experiment configuration from")
    parser.add_argument("-n", type=int, default=None, help="Number of to be evaluated samples")

    args = parser.parse_args()

    with open(args.path) as file:
        config = yaml.safe_load(file)
  
    print(f"EXPERIMENT: {config['description']}")
    
    flat_config = {}
    for entry in config:
        if entry != 'description':
            if config[entry] =="None":
                config[entry] = None
            flat_config.update({entry: config[entry]})

    if args.n is not None:
        flat_config["number_verifications"] = args.n
    print(flat_config)

    flat_config = Bunch(flat_config)
    verify(flat_config)