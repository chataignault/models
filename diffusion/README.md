# Diffusion models

> [!NOTE]
> Each subfolder contains environment configuration


## Scripts

> [!NOTE]
> Requirements in *test_requirements.txt*

**Animate Diff**

```bash
uv run test_animate_diff.py --prompt "sea-side, beach, high resolution, pine trees, waves"
```


**Stable Diffusion**

```bash
uv run test_stable_diffusion.py --prompt "Jacques Prévert en maillot de bain"
```

> [!CAUTION]
> Prompt should be in English

**Debug bash**

If UTF-8 is not supported by default, one can specify the following variables :

```bash
# export LC_ALL=en_UK.UTF-8
export LANG=en_UK.UTF-8
export LANGUAGE=en_UK.UTF-8
```

**To go further...**

![image](img/sea_side_beach.gif)

![image](img/jp_bowtie.png)

