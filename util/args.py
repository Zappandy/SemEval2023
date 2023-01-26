import argparse

def create_arg_parser():

    """Returns a map with commandline parameters taken from the user"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-lan", "--language", default='en', type=str, help="Provide language"
    )

    parser.add_argument(
        "-m", "--model", default='distilbert-base-uncased', type=str, help="Provide model and tokenizer"
    )

    parser.add_argument(
        "-s", "--set", default=None, type=str, help="Provide setup"
    )

    parser.add_argument(
        "-crf", "--crf", default=False, type=bool, help="true or false"
    )

    parser.add_argument(
        "-ss", "--save_steps", default=1000, type=int, help="Provide number of save steps"
    )
    
    parser.add_argument(
        "-b", "--batch_size", default=12, type=int, help="Change batch size"
    )

    args = parser.parse_args()
    return args
