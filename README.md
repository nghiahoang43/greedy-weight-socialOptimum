


**Project Name**

**Installation**
This project requires Python and some dependencies that are listed in requirements.txt.

**Step 1: Install Python**
If you don't have Python installed, you can download it from the official website:

https://www.python.org/downloads/

**Step 2: Clone the repository**
Clone this repository to your local machine. You can do this by running the following command in your terminal:

```
git clone https://github.com/nghiahoang43/greedy-weight-socialOptimum.git
```

***Step 3: Install the dependencies***
Navigate to the directory of the cloned repository. This is where the requirements.txt file is located. You can navigate to this directory by running:

```
cd greedy-weight-socialOptimum
```

Now, install the dependencies by running:

```
pip install -r requirements.txt
```
This command will install all the dependencies listed in requirements.txt.

Running the Project
After installing the dependencies, you can run the project by executing:
```
python er.py
```

Copy this code block to compare utility:
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify internal regret settings')
    parser.add_argument('--num_moves', type=int, default=4, help="Number of moves in the game, default 3")
    parser.add_argument('--num_players', type=int, default=4, help="Number of players in the game, default 2")
    parser.add_argument('--iterations', type=int, default=10000, help="Number of iterations to run regret minimization, default 1K")
    parser.add_argument('--seed', type=int, default=42, help="Random seed, default 42")
    parser.add_argument('--zero_sum', action="store_true", default=False, help="Make the game zero sum, default false")
    parser.add_argument('--solo_compare', action="store_true", default=False, help="Compare only RM and dynamic weights")
    parser.add_argument('--sweep_floor', action="store_true", default=False, help="Sweep floors")
    parser.add_argument('--compare_dynamic', action="store_true", default=False, help="Compare the variants of dynamic weights")
    parser.add_argument('--full_sweep', action="store_true", default=False, help="Sweep through all non dynamic methods")
    parser.add_argument('--full_utility', action="store_true", default=True, help="Sweep through all non dynamic methods")
    parser.add_argument('--compare_mixed', action="store_true", default=False, help="Compare mixed dynamic")
    parser.add_argument('--use_time', action="store_true", default=False, help="Compare time")
    parser.add_argument('--pure', action="store_true", default=True, help="Use pure strategies for regret minimization, default false")
    parser.add_argument("--experiment_name", type=str, help="What is this experiment called?", default="CYBERTRASH")
    parser.add_argument("--game_reps", type=int, help="How many games to repeat for", default=1)
    args = parser.parse_args()
    main(args.num_players, args.num_moves, args.zero_sum, args.seed, args.iterations, args.pure, args.experiment_name, args.game_reps)
```