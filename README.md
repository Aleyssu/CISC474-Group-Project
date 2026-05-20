# Gridworld Reinforcement Learning Agent
A reinforcement learning agent implemented in stable diffusion to fully explore a grid-based map with predefined enemies and obstacles.

Install required packages with
```bash
pip install -r requirements.txt
```

Install coverage-gridworld with
```
pip install -e coverage-gridworld
```

Simply run `main.py` to see the agent play through the predefined maps. Feel free to modify the maps in the script as you please.

If you wish to use any of the unused models, you'll need to replace the `custom.py` in `coverage_gridworld/` with the `custom.py` of the model you wish to use and load the given model. If you've installed coverage-gridworld with the `-e` parameter you shouldn't have to reinstall it and you should be ready to go. Just make sure you're loading the correct model in `main.py`.
