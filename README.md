# APR Benchmark Results Repository

## Repo structure

The main structure of the repo is:

```
- data/
    - <benchmark>/
        - <model>/
            - <generated_data_type>/
                - data_1.jsonl
                - ...
            - sample_file.jsonl
```

There are also multiply .ipynb files used to analyze and transform the repo data:

- `combine_data.ipynb` is used to combine multiple files of genreated data into a single file, used when it is not possible to generate a single file with enough samples due to GPU constraints.
- `data_leakage_experiments.ipynb` is used to calculate the CDD & TED score of generated data. 
- `gemeni_analysis.ipynb` is used to find repositories of potential leaked data used as training data, and perform analysis of these repositories.
