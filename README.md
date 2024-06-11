# Email Subject Classifier

This is simble C# console application uses Microsoft ML.NET to classify email subjects into different departments. It trains a model using a dataset of email subjects and their corresponding departments, and then uses this model to predict the department for new email subjects.

## Features

- Train a machine learning model to classify email subjects.
- Save and load the trained model.
- Predict the department for new email subjects interactively.

## Prerequisites

- .NET Core SDK
- Microsoft ML.NET

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Tarek-Qa/EmailSubjectClassifire.git
    cd EmailSubjectClassifier
    ```

2. Install the required .NET packages:
    ```bash
    dotnet add package Microsoft.ML
    ```

## Usage

1. Prepare your training data in a TSV file with headers (`Subject` and `Department`).
2. Update the file paths in the code:
    ```csharp
    string _trainingFilePath = "path/to/your/SubjectsModel.tsv";
    ```

3. Run the application:
    ```bash
    dotnet run
    ```

4. Follow the prompts in the console to predict the department for new email subjects.


