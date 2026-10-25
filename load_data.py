from datasets import load_dataset

from consts import (
    HF_ANLI_PATH,
    HF_ANLI_SPLIT_NAME,
    HF_CASEHOLD_SUBSET_NAME,
    HF_HELLASWAG_PATH,
    HF_JEV_TYPED_DECISION_PATH,
    HF_LEDGAR_PATH,
    HF_LEDGAR_SUBSET_NAME,
    HF_MMLU_PATH,
    HF_MMLU_SUBSET_NAME,
    HF_NLI_PATH,
    HF_VITAMIN_PATH,
    HF_NLI_REVISION_NAME,
    HF_OPEN_JEV_PATH,
    HF_SCIQ_PATH,
    MAX_CHARS,
    N_TRAIN_SAMPLES_JEV_PER_DATASET,
    N_TRAIN_SAMPLES_PER_DATASET,
    N_VALID_SAMPLES_PER_DATASET,
    TRAIN_SPLIT_NAME,
    VAL_SPLIT_NAME,
)
from data import ds_sample, filter_jev


def load_train():
    data_anli_train = ds_sample(
        load_dataset(HF_ANLI_PATH, split=HF_ANLI_SPLIT_NAME),
        N_TRAIN_SAMPLES_PER_DATASET,
    )
    data_mmlu_train = ds_sample(
        load_dataset(HF_MMLU_PATH, HF_MMLU_SUBSET_NAME, split=TRAIN_SPLIT_NAME),
        N_TRAIN_SAMPLES_PER_DATASET,
    )
    data_sciq_train = ds_sample(
        load_dataset(HF_SCIQ_PATH, split=TRAIN_SPLIT_NAME), N_TRAIN_SAMPLES_PER_DATASET
    )
    data_hellaswag_train = ds_sample(
        load_dataset(HF_HELLASWAG_PATH, split=TRAIN_SPLIT_NAME),
        N_TRAIN_SAMPLES_PER_DATASET,
    )

    data_tasksource_train = ds_sample(
        filter_jev(
            load_dataset(HF_JEV_TYPED_DECISION_PATH, split=TRAIN_SPLIT_NAME),
            "state",
            MAX_CHARS,
        ),
        N_TRAIN_SAMPLES_JEV_PER_DATASET,
    )
    data_openjev_train = ds_sample(
        filter_jev(
            load_dataset(HF_OPEN_JEV_PATH, split=TRAIN_SPLIT_NAME),
            "state_json",
            MAX_CHARS,
        ),
        N_TRAIN_SAMPLES_JEV_PER_DATASET,
    )
    data_ledgar_train = ds_sample(
        load_dataset(HF_LEDGAR_PATH, HF_LEDGAR_SUBSET_NAME, split=TRAIN_SPLIT_NAME),
        N_TRAIN_SAMPLES_PER_DATASET,
    )
    data_casehold_train = ds_sample(
        load_dataset(HF_LEDGAR_PATH, HF_CASEHOLD_SUBSET_NAME, split=TRAIN_SPLIT_NAME),
        N_TRAIN_SAMPLES_PER_DATASET,
    )
    data_nli_train = ds_sample(
        load_dataset(
            HF_NLI_PATH, revision=HF_NLI_REVISION_NAME, split=TRAIN_SPLIT_NAME
        ),
        N_TRAIN_SAMPLES_PER_DATASET,
    )

    data_vitamin_train = ds_sample(
        load_dataset(
            HF_VITAMIN_PATH, split=TRAIN_SPLIT_NAME
        ),
        N_TRAIN_SAMPLES_PER_DATASET,
    )

    data_mmlu_train = data_mmlu_train.filter(
        lambda r: len(set(r["train"]["choices"])) == len(r["train"]["choices"])
    )
    data_sciq_train = data_sciq_train.filter(
        lambda r: (
            len(
                {
                    r["distractor1"],
                    r["distractor2"],
                    r["distractor3"],
                    r["correct_answer"],
                }
            )
            == 4
        )
    )
    data_nli_train = data_nli_train.filter(lambda r: len(r["premise"]) < MAX_CHARS)
    data_vitamin_train = data_vitamin_train.filter(lambda r: len(r["evidence"]) < MAX_CHARS)

    return (
        data_anli_train,
        data_mmlu_train,
        data_sciq_train,
        data_hellaswag_train,
        data_tasksource_train,
        data_openjev_train,
        data_ledgar_train,
        data_casehold_train,
        data_nli_train,
        data_vitamin_train,
    )


def load_valid():
    data_tasksource_valid = load_dataset(
        HF_JEV_TYPED_DECISION_PATH, split=VAL_SPLIT_NAME
    )
    data_openjev_valid = load_dataset(HF_OPEN_JEV_PATH, split=VAL_SPLIT_NAME)
    data_ledgar_valid = load_dataset(
        HF_LEDGAR_PATH, HF_LEDGAR_SUBSET_NAME, split=VAL_SPLIT_NAME
    )
    data_casehold_valid = load_dataset(
        HF_LEDGAR_PATH, HF_CASEHOLD_SUBSET_NAME, split=VAL_SPLIT_NAME
    )
    data_nli_valid = load_dataset(
        HF_NLI_PATH, revision=HF_NLI_REVISION_NAME, split=VAL_SPLIT_NAME
    )
    data_hellaswag_valid = load_dataset(HF_HELLASWAG_PATH, split=VAL_SPLIT_NAME)

    data_tasksource_valid = ds_sample(
        filter_jev(data_tasksource_valid, "state", MAX_CHARS),
        N_VALID_SAMPLES_PER_DATASET,
    )
    data_openjev_valid = ds_sample(
        filter_jev(data_openjev_valid, "state_json", MAX_CHARS),
        N_VALID_SAMPLES_PER_DATASET,
    )
    data_ledgar_valid = ds_sample(data_ledgar_valid, N_VALID_SAMPLES_PER_DATASET)
    data_casehold_valid = ds_sample(data_casehold_valid, N_VALID_SAMPLES_PER_DATASET)
    data_nli_valid = data_nli_valid.filter(lambda r: len(r["premise"]) < MAX_CHARS)
    data_nli_valid = ds_sample(data_nli_valid, N_VALID_SAMPLES_PER_DATASET)
    data_hellaswag_valid = ds_sample(data_hellaswag_valid, N_VALID_SAMPLES_PER_DATASET)

    return (
        data_tasksource_valid,
        data_openjev_valid,
        data_ledgar_valid,
        data_casehold_valid,
        data_nli_valid,
        data_hellaswag_valid,
    )
