from flask_wtf import FlaskForm
from wtforms import StringField, RadioField, SubmitField, validators
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms.validators import DataRequired, Length



class textForm(FlaskForm):
    text = StringField("Text here", validators=[DataRequired(), Length(min=4, max=200)])
    model = RadioField('model', choices=[
        ('sequential', 'Sequential'),
        ('logistic', 'Logistic'),
        ('svm', 'SVM'),
        ('all_models', 'All Models'),
    ], validators=[validators.Optional()], default='svm')
    submit = SubmitField("Click To Predict")




class FileForm(FlaskForm):
    csv_file = FileField(
        'Upload CSV File',
        validators=[
            FileRequired(),
            FileAllowed(['csv'], 'Only CSV files are allowed.')
        ]
    )

    model = RadioField('model', choices=[
        ('sequential', 'Sequential'),
        ('logistic', 'Logistic'),
        ('svm', 'SVM'),
        ('all_models', 'All Models'),
    ], validators=[validators.Optional()], default='svm')

    submit = SubmitField("Click To Predict")
