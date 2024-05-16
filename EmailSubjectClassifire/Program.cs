using Microsoft.ML;
using Microsoft.ML.Data;


string _traningFilePath = "C:\\Users\\Tarek\\source\\repos\\EmailSubjectClassifire\\EmailSubjectClassifire\\Models\\SubjectsModel.tsv";
string _modelFilePath = "C:\\Users\\Tarek\\source\\repos\\EmailSubjectClassifire\\EmailSubjectClassifire\\Models\\model.zip";
MLContext _mLcontext;
IDataView _trainingDataView;
ITransformer _model;

PredictionEngine<Emailsubject, DepartmentPrediction> _predictionEngin;

_mLcontext = new MLContext(seed: 0);
_trainingDataView = _mLcontext.Data.LoadFromTextFile<Emailsubject>(_traningFilePath, hasHeader: true);
var pipeline = ProcessData();
var trainingPipeline = BildAndTrainModel(_trainingDataView, pipeline);
SaveModleAsFile();
var result = PredictDepartmintForSubjectLine("New Invoice");


var KeepRunning = true;
Console.WriteLine("Enter subject line to predict. Type QUIT to close the app");
while (KeepRunning)
{
    var subjectLine = Console.ReadLine();
    if (subjectLine == "QUIT")
    {
        KeepRunning = false;
    }
    else
    {
        Console.WriteLine(PredictDepartmintForSubjectLine(subjectLine));
    }
}
string PredictDepartmintForSubjectLine(string subjectLine)
{
    var model = _mLcontext.Model.Load(_modelFilePath, out var modelInputSchema);
    var emailSubject = new Emailsubject() { Subject = subjectLine};
    _predictionEngin = _mLcontext.Model.CreatePredictionEngine<Emailsubject, DepartmentPrediction>(model);
    var resoult = _predictionEngin.Predict(emailSubject);
    return resoult.Department;
}
void SaveModleAsFile()
{
    _mLcontext.Model.Save(_model, _trainingDataView.Schema, _modelFilePath);
}
IEstimator<ITransformer> ProcessData()
{
    var Pipeline = _mLcontext.Transforms.Conversion.MapValueToKey(inputColumnName: "Department", outputColumnName: "Label")
            .Append(_mLcontext.Transforms.Text.FeaturizeText(inputColumnName: "Subject", outputColumnName: "EmailSubjectFeaturized"))
           .Append(_mLcontext.Transforms.Concatenate("Features", "EmailSubjectFeaturized"))
            .AppendCacheCheckpoint(_mLcontext);
    return Pipeline;
}

IEstimator<ITransformer>BildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
{
    var trainingPipeline = pipeline.Append(_mLcontext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
           .Append(_mLcontext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
    _model = trainingPipeline.Fit(trainingDataView);
    return pipeline;
}
public class Emailsubject
{
    [LoadColumn(0)]
    public string Subject { get; set; }

    [LoadColumn(1)]
    public string Department { get; set; }
}

public class DepartmentPrediction
{
    [ColumnName("PredictedLabel")]
    public string? Department { get; set; }

}
