from datasets import DatasetDict,load_dataset

dataset = load_dataset(
    "parquet",
    data_files={
        "train": "comprehensive-qa-dataset/data/train-00000-of-00001.parquet",
        "validation": "comprehensive-qa-dataset/data/validation-00000-of-00001.parquet"
    }
)
# Funkcja konwertująca jeden split do formatu [{"input":..., "output":...}, ...]
def convert_to_input_output(ds: DatasetDict, question_col="question", answer_col="answer"):
    result = []
    for split in ds.keys():  # np. "train" i "validation"
        for q, a in zip(ds[split][question_col], ds[split][answer_col]):
            # answer może być listą, jeśli dataset QA ma wiele odpowiedzi
            if isinstance(a, list):
                # weź pierwszą odpowiedź, jeśli jest lista
                a = a[0] if len(a) > 0 else ""
            result.append({"input": q, "output": a})
    return result

# Konwersja
converted_data = convert_to_input_output(dataset)

# Sprawdź kilka pierwszych
for x in converted_data[:5]:
    print(x)
