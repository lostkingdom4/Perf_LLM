# Perf_LLM
This code is used for MLSYS submission.

The Framework use RL to train a LM to improve the source code execution time.

As our model is trained on a online server, the cpu of online server is not stable to test the execution time. We need to do execution time estimation in our own server.

Work Flow:
1. bash init_m.sh: Generate dataset for train, validation, & test. Measure the corresponding execution time.
2. generate_T5.sh(remote server): random sampling codes, and save to a list in ./generated_list/
3. measurement.sh(local server): measure the run time speed with the reward function. Generate a dataset with all the required information for computing loss to ./RRHF_output/output_dataset.pt
4. RL_T5.sh: trainer.

Repeat step 2-4 to do more RL steps.
