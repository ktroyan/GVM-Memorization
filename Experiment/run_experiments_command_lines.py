import itertools
import subprocess


# Experiment 1. Real artworks to real artworks and gen artworks to gen artworks
sd_version = ["1.5", "2.1"]
clip_version = ["openai", "laion2b"]
image_realm = ["real", "gen"]
artist_category = ["historical", "artstation"]

combinations = list(itertools.product(sd_version, clip_version, image_realm, artist_category))

for combination in combinations:
    sd_version_val, clip_version_val, image_realm_val, artist_category_val = combination
    if image_realm_val == "real" and sd_version_val == "1.5":
        continue
    command = f"python ./Experiment/experiment.py --sd_version {sd_version_val} --clip_version {clip_version_val} --image_realm {image_realm_val} --artist_category {artist_category_val} --eval_mode --topk 5 --use_tqdm --balance_classes_across_sets --rand_shuffle_data --multi_top_k --save_experiment_results --pred_model logistic_regression"
    subprocess.run(command, shell=True)


# Experiment 2. Different realms. Real artworks to gen artworks and gen artworks to real artworks
sd_version = ["1.5", "2.1"]
clip_version = ["openai", "laion2b"]
image_realm = ["real", "gen"]
artist_category = ["historical", "artstation"]

combinations = list(itertools.product(sd_version, clip_version, image_realm, artist_category))

for combination in combinations:
    sd_version_val, clip_version_val, image_realm_val, artist_category_val = combination
    command = f"python ./Experiment/experiment.py --sd_version {sd_version_val} --clip_version {clip_version_val} --image_realm {image_realm_val} --artist_category {artist_category_val} --eval_mode --topk 5 --use_tqdm --balance_classes_across_sets --rand_shuffle_data --multi_top_k --save_experiment_results --pred_model logistic_regression --train_test_diff_realm"
    print(command)
    subprocess.run(command, shell=True)


# Experiment 3. Merged realms. Real+Gen artworks to Real+Gen artworks
sd_version = ["1.5", "2.1"]
clip_version = ["openai", "laion2b"]
image_realm = ["merged"]
artist_category = ["historical", "artstation"]

combinations = list(itertools.product(sd_version, clip_version, image_realm, artist_category))

for combination in combinations:
    sd_version_val, clip_version_val, image_realm_val, artist_category_val = combination
    command = f"python ./Experiment/experiment.py --sd_version {sd_version_val} --clip_version {clip_version_val} --image_realm {image_realm_val} --artist_category {artist_category_val} --eval_mode --topk 5 --use_tqdm --balance_classes_across_sets --rand_shuffle_data --multi_top_k --save_experiment_results --pred_model logistic_regression"
    subprocess.run(command, shell=True)