"""
# Author Information
======================
Author: Cedric Manouan & Arisema Mihretu
Last Update: 18 Oct, 2023
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--generate_file", 
    type=bool, 
    default=False, 
    help="wheather to generate vocab file or not."
)

parser.add_argument(
    "--vocab_dir", 
    type=str, 
    default="../data/", 
    help="Directory containing vocabulary file(s)."
)

ACTION_DESCRIPTION = [
    'forwards', 
    'move', 
    'in', 
    'put', 
    'to', 
    'the', 
    'top', 
    'behind', 
    'right', 
    'of', 
    'on', 
    'shift', 
    'left', 
    'front', 
    'backwards'
    ]

OBJECTS = [
    ':POT', 
    ':BLUE-METAL-PLATE', 
    ':SPOON', 
    ':SHOE', 
    ':FORK', 
    ':BOTTLE', 
    ':GLASSES', 
    ':SPATULA',  
    ':CEREAL', 
    ':WEISSWURST', 
    ':BREAKFAST-CEREAL', 
    ':GLOVE', 
    ':BUTTERMILK', 
    ':RED-METAL-PLATE',     
    ':KNIFE', 
    ':MONDAMIN', 
    ':MUG', 
    ':CAP', 
    ':BOWL',     
    ':CUBE', 
    ':MILK', 
    ':PLATE', 
    ':CUP', 
    ':TRAY'   
]


MOTOR_COMMANDS = [
    'POSE-4', 
    'POSE-8', 
    ':POT', 
    'POSE-10', 
    'RED', 
    'POSE-11', 
    'POSE-15', 
    'POSE-5', 
    'POSE-12', 
    'POSE-3', 
    'POSE-14', 
    'NIL', 
    'POSE-9', 
    'POSE-2', 
    'GREEN', 
    "#'*on-transformation*", 
    "#'*leftward-transformation*", 
    "#'*forward-transformation*", 
    "#'*backward-transformation*", 
    'BLUE', 
    'POSE-6', 
    "#'*rightward-transformation*", 
    'POSE-13', 
    'POSE-7', 
    'POSE-1', 
    ]

SPECIAL_TOKENS = ["[SOS]", "[PAD]", "[UNK]", "[EOS]"] 


if __name__ == '__main__':
    
    args = parser.parse_args()
    
    if args.generate_file:
        ALL_TOKENS = set(SPECIAL_TOKENS+ OBJECTS + ACTION_DESCRIPTION + MOTOR_COMMANDS)
        
        with open("../data/simpleTokenizer.txt", "w") as v:
            for t in sorted(ALL_TOKENS):
                v.write(t+"\n")    
    else:
        with open("../data/simpleTokenizer.txt", "r") as v:
            ALL_TOKENS = [t.strip() for t in v.readlines()]

        TOKENS_MAPPING = {t:i for i, t in enumerate(ALL_TOKENS)}
        REVERSE_TOKENS_MAPPING = {i:t for i,t in enumerate(ALL_TOKENS)}

        print("# action tokens: \t", len(ACTION_DESCRIPTION))
        print("# command tokens: \t", len(MOTOR_COMMANDS))
        print("# tokens in total: \t", len(ALL_TOKENS))
