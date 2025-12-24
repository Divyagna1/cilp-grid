# causal-gym
causal-gym

## Installation
At the project root directory, type in 
```
pip install .
```

## Windy Lava Grid

The repository now provides a configurable `WindyLavaGridEnv` under `causal_gym.envs`.  
You can drop it into existing notebooks exactly like the original Windy gridworld while
customizing:
- Grid dimensions, agent start, and target locations.
- Sets of lava cells plus their penalties and terminal behavior.
- Random gust distributions together with deterministic column wind.
- Default behavior policies (callable or tensor-valued) for SCM interventions.
