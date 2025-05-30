# mlutils
My ml utils(mostly IL/RL) that I keep reusing over the year, refactored.

## Requirements
The requirements are `hydra`, `numpy`, `matplotlib` and `pytorch` and the rest is optional.

Most of the code expects certain structure, most notably on the `hydra` config.
It expect it to contain (at the root-level) the structure similar to
```
defaults:
    - _self_
    - otherstuff: otherstuff

eval:
    render: False # True/False
#If wandb used
wandb:
    project: "project-name"
    entity: "user or team name"
    name:
    group:

hydra:
    job:
        chdir: true # it's expected that each run is done on a different directory
    run:
        dir: ./some/folder/that/changes/name/${now:%m-%d_%H-%M-%S}
```
