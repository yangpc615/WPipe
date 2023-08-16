import os
import json
import numpy as np

cv_config_file = "throughput{}x{}cv.json"
cv_config_file_h = "throughput{}x{}cvh.json"
nlp_config_file = "throughput{}x{}nlp.json"

stage_str = "stage{}"
sub_config = {"model": None, "parallel": None}
template = {"network_config": None}


def generate_cv_config(result,
                       model_parallel_size,
                       data_parallel_size,
                       is_heterogeneous=False):
    num_gpus = model_parallel_size * data_parallel_size
    cur_config = []
    for i in range(model_parallel_size):
        stage = stage_str.format(i)
        cur_sub_config = sub_config.copy()
        cur_sub_config["model"] = [i, i + model_parallel_size]
        cur_sub_config["parallel"] = np.arange(i, num_gpus,
                                               model_parallel_size).tolist()
        if is_heterogeneous:
            cur_sub_config["parallel"] = np.arange(i * data_parallel_size,
                                       (i + 1) * data_parallel_size).tolist()
        cur_config.append([stage, cur_sub_config])
    result["network_config"] = cur_config
    return result


def generate_nlp_config(result, model_parallel_size, data_parallel_size):
    cur_config = []
    for i in range(model_parallel_size):
        stage = stage_str.format(i)
        cur_sub_config = sub_config.copy()
        cur_sub_config["model"] = [i, i + model_parallel_size]
        cur_sub_config["parallel"] = np.arange(
            i * data_parallel_size, (i + 1) * data_parallel_size).tolist()
        cur_config.append([stage, cur_sub_config])
    result["network_config"] = cur_config
    return result


def generate_config(num_gpus,
                    model_parallel_size,
                    file_name_tpl,
                    task,
                    is_heterogeneous=False):
    while (num_gpus // model_parallel_size >= 1):
        data_parallel_size = num_gpus // model_parallel_size
        file_name = file_name_tpl.format(model_parallel_size,
                                         data_parallel_size)
        cur_template = template.copy()
        if task == "cv":
            result = generate_cv_config(cur_template, model_parallel_size,
                                        data_parallel_size, is_heterogeneous)
        elif task == "nlp":
            result = generate_nlp_config(cur_template, model_parallel_size,
                                         data_parallel_size)
        else:
            raise ValueError(f"The task {task} not exist!")
        with open(os.path.join("./network_conf", file_name), "w") as f:
            json.dump(result, f)
        model_parallel_size *= 2


if __name__ == "__main__":
    num_gpus = 64

    nlp_min_split = 8
    cv_min_split = 4
    # generate_config(num_gpus, nlp_min_split, nlp_config_file, "nlp")
    generate_config(num_gpus, cv_min_split, cv_config_file_h, "cv", True)
    generate_config(num_gpus, cv_min_split, cv_config_file, "cv", False)
