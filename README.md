# Python Easy Text Classification Model

This repository contains a text classification model implemented in Python using the `scikit-learn` library. The model classifies text into "good" or "bad" categories based on training data provided in CSV files.

Tested on Python 3.12.1
## Usage

### Options

The model supports various options specified in a JSON format:

Example:
```json
{
    "modelname": "textclassification",
    "hashname": "csvhashes",
    "checkCSVfiles": "true",
    "inputMode": "true",
    "enablePrints": "true",
    "stringJSONreply": "false"
}
```

- `modelname`: Name of the trained model file. If you keep default, model file wil be `textclassification.joblib`
- `hashname`: Name of the file storing the hash of the concatenated CSV data.
- `checkCSVfiles`: Set to "true" to ensure CSV files have the required number of rows. (min. 2)
- `inputMode`: Set to "true" for interactive user input, otherwise uses predefined text.
- `enablePrints`: Set to "true" to enable print statements.
- `stringJSONreply`: Set to "true" to get JSON-formatted string prediction output.

### Training and Usage

1. **Training the Model:**
   - The code will train the model using data from `good_texts.csv` and `bad_texts.csv`.
   - Data hash is calculated to determine if retraining is needed.

2. **User Input Mode:**
   - If `inputMode` is set to `true`, the user can input text for classification interactively.

3. **Output:**
   - The model outputs predictions and probabilities for "good" and "bad" classes.

## Dependencies

- pandas
- scikit-learn
- joblib
- hashlib

To install:
```pip install pandas scikit-learn joblib hashlib```

## Getting Started

1. Clone this repository.
2. Ensure Python and required libraries are installed. (Currently tested in Python 3.12.1)
3. Customize options and CSV files in the code or use default settings.
4. Run the script.

Feel free to explore and modify the code for your specific use case. If you encounter any issues or have suggestions, don't forget open an issue.
